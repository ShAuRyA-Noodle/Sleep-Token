"""
GPU-accelerated Monte Carlo simulation for SupplyMind.

Expands a single state to a batch of 100K, adds noise scaled by
linspace(0.01, 0.3), runs one GPU pass through the surrogate,
and returns the full distribution.

Target: <80ms for 100K scenarios on RTX 4080.

Returns: p5, p50, p95, p99, cvar_10, full distribution for violin plot.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class GPUMonteCarlo:
    """GPU-accelerated Monte Carlo scenario simulation.

    Takes a single state + action, expands to 100K variants with
    different noise levels, runs them all through the neural surrogate
    in one GPU pass, returns distribution statistics.

    Args:
        model:         Trained WorldModel (or None to load from checkpoint).
        n_scenarios:   Number of scenarios to simulate (default 100_000).
        noise_min:     Minimum noise scale (default 0.01).
        noise_max:     Maximum noise scale (default 0.3).
        device:        Torch device.
    """

    def __init__(
        self,
        model=None,
        n_scenarios: int = 100_000,
        noise_min: float = 0.01,
        noise_max: float = 0.3,
        device: str = "cuda",
    ) -> None:
        self.n_scenarios = n_scenarios
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.device = device
        self.model = model

        # Pre-compute noise scales: linspace from noise_min to noise_max
        self._noise_scales = torch.linspace(
            noise_min, noise_max, n_scenarios, device=device,
        ).unsqueeze(1)  # (N, 1)

    def load_model(self) -> None:
        """Load world model from checkpoint if not provided."""
        if self.model is not None:
            return
        from rl.surrogate.world_model import load_world_model
        self.model = load_world_model(device=self.device)

    @torch.no_grad()
    def simulate(
        self,
        state: np.ndarray,
        action_flat: int,
        total_revenue: float = 1e9,
        horizon: int = 5,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation from a single state.

        Args:
            state:          (408,) state vector.
            action_flat:    Flat action index (0-279).
            total_revenue:  For dollar conversion.
            horizon:        Number of steps to simulate (default 5).

        Returns:
            Dict with: p5, p50, p95, p99, cvar_10, mean, std,
                       distribution (full array for violin plot),
                       elapsed_ms.
        """
        self.load_model()
        start = time.perf_counter()

        state_t = torch.from_numpy(state).float().to(self.device)
        N = self.n_scenarios

        # Expand state to batch
        state_batch = state_t.unsqueeze(0).expand(N, -1).clone()

        # Add scaled noise
        noise = torch.randn(N, 408, device=self.device)
        state_batch = state_batch + noise * self._noise_scales

        # Action one-hot
        action_oh = torch.zeros(N, 280, device=self.device)
        action_oh[:, min(action_flat, 279)] = 1.0

        # Roll out horizon steps, accumulating reward
        cumulative_rewards = torch.zeros(N, device=self.device)

        for t in range(horizon):
            next_state, reward, done_prob = self.model(state_batch, action_oh)
            cumulative_rewards += reward.squeeze(-1)
            state_batch = next_state

            # After first step, switch to do_nothing
            if t == 0:
                action_oh = torch.zeros(N, 280, device=self.device)
                action_oh[:, 0] = 1.0  # do_nothing

        # Convert to numpy for percentile computation
        rewards_np = cumulative_rewards.cpu().numpy()

        # Dollar conversion
        reward_to_dollar = total_revenue / 100
        losses_dollars = -rewards_np * reward_to_dollar  # negative reward = loss

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Compute statistics
        result = {
            "p5": float(np.percentile(losses_dollars, 5)),
            "p50": float(np.percentile(losses_dollars, 50)),
            "p95": float(np.percentile(losses_dollars, 95)),
            "p99": float(np.percentile(losses_dollars, 99)),
            "mean": float(np.mean(losses_dollars)),
            "std": float(np.std(losses_dollars)),
            "n_scenarios": N,
            "horizon": horizon,
            "elapsed_ms": round(elapsed_ms, 1),
        }

        # CVaR at 10% (expected loss in worst 10% of scenarios)
        k = max(1, int(0.1 * N))
        sorted_losses = np.sort(losses_dollars)[::-1]  # descending (worst first)
        result["cvar_10"] = float(np.mean(sorted_losses[:k]))

        # Full distribution (downsampled for violin plot — keep 1000 points)
        indices = np.linspace(0, N - 1, min(1000, N), dtype=int)
        result["distribution"] = losses_dollars[indices].tolist()

        if elapsed_ms < 80:
            logger.info("GPU MC: %d scenarios in %.1fms (target: <80ms) - PASS", N, elapsed_ms)
        else:
            logger.warning("GPU MC: %d scenarios in %.1fms (target: <80ms) - SLOW", N, elapsed_ms)

        return result


def main() -> None:
    """Quick benchmark of GPU Monte Carlo."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    mc = GPUMonteCarlo(n_scenarios=100_000, device="cuda")
    mc.load_model()

    # Dummy state
    state = np.random.randn(408).astype(np.float32)
    result = mc.simulate(state, action_flat=42, total_revenue=1e9)

    print(f"P5:   ${result['p5']:,.0f}")
    print(f"P50:  ${result['p50']:,.0f}")
    print(f"P95:  ${result['p95']:,.0f}")
    print(f"P99:  ${result['p99']:,.0f}")
    print(f"CVaR: ${result['cvar_10']:,.0f}")
    print(f"Time: {result['elapsed_ms']}ms for {result['n_scenarios']:,} scenarios")


if __name__ == "__main__":
    main()
