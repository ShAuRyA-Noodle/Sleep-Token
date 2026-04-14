"""
Counterfactual engine for SupplyMind.

After each action, replays the episode with do_nothing from that point
using the neural surrogate world model. Outputs the difference:
"Without this backup activation, P50 additional loss: $X"

This is the "what-if" analytical backbone: for every agent decision,
we can quantify exactly how much value it added vs. inaction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class CounterfactualEngine:
    """Computes counterfactual outcomes using the neural surrogate.

    For a given (state, action_taken), simulates what would have happened
    if the agent had done nothing instead, projecting forward N steps.

    Args:
        model:          Trained WorldModel instance.
        horizon:        Number of steps to project forward (default 10).
        n_simulations:  Number of stochastic forward passes for CI (default 50).
        device:         Torch device.
    """

    def __init__(
        self,
        model=None,
        horizon: int = 10,
        n_simulations: int = 50,
        device: str = "cuda",
    ) -> None:
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.device = device
        self.model = model

    def load_model(self) -> None:
        """Load world model from checkpoint if not provided."""
        if self.model is not None:
            return
        from rl.surrogate.world_model import load_world_model
        self.model = load_world_model(device=self.device)

    @torch.no_grad()
    def compute_counterfactual(
        self,
        state: np.ndarray,
        action_taken: int,
        total_revenue: float = 1e9,
    ) -> dict[str, Any]:
        """Compute counterfactual: what if we did nothing instead?

        Args:
            state:          (408,) current state vector.
            action_taken:   Flat action index that was actually taken.
            total_revenue:  Total revenue at risk (for dollar conversion).

        Returns:
            Dict with:
              - reward_with_action: expected reward trajectory with action
              - reward_without_action: expected reward trajectory with do_nothing
              - additional_loss_p50: P50 additional loss from inaction ($)
              - additional_loss_p95: P95 additional loss from inaction ($)
              - value_of_action: how much value the action added ($)
              - explanation: human-readable summary
        """
        self.load_model()

        state_t = torch.from_numpy(state).float().to(self.device)

        # Action one-hot
        action_oh = torch.zeros(280, device=self.device)
        action_oh[min(action_taken, 279)] = 1.0

        # Do-nothing one-hot (action type 0 = do_nothing, node 0)
        noop_oh = torch.zeros(280, device=self.device)
        noop_oh[0] = 1.0

        # --- Simulate WITH action taken ---
        rewards_with = self._rollout(state_t, action_oh, noop_oh)

        # --- Simulate WITHOUT action (do_nothing from start) ---
        rewards_without = self._rollout(state_t, noop_oh, noop_oh)

        # Compute cumulative rewards
        cum_with = np.cumsum(rewards_with, axis=1)
        cum_without = np.cumsum(rewards_without, axis=1)

        # Additional loss = difference in cumulative reward (negative = worse)
        loss_diff = cum_without[:, -1] - cum_with[:, -1]  # positive means action helped

        # Convert to dollars
        reward_to_dollar = total_revenue / 100  # approximate scaling
        loss_diff_dollars = loss_diff * reward_to_dollar

        p50_loss = float(np.percentile(loss_diff_dollars, 50))
        p95_loss = float(np.percentile(loss_diff_dollars, 95))
        value_of_action = float(np.mean(loss_diff_dollars))

        # Action type name
        action_type_idx = action_taken // 40
        action_types = [
            "do_nothing", "activate_backup_supplier", "reroute_shipment",
            "increase_safety_stock", "expedite_order", "hedge_commodity",
            "issue_supplier_alert",
        ]
        action_name = action_types[min(action_type_idx, 6)]

        explanation = (
            f"Without this {action_name}, "
            f"P50 additional loss: ${abs(p50_loss):,.0f}. "
            f"Action value: ${value_of_action:,.0f} "
            f"(P95 worst case: ${abs(p95_loss):,.0f})."
        )

        return {
            "reward_with_action": rewards_with.mean(axis=0).tolist(),
            "reward_without_action": rewards_without.mean(axis=0).tolist(),
            "additional_loss_p50": p50_loss,
            "additional_loss_p95": p95_loss,
            "value_of_action": value_of_action,
            "explanation": explanation,
            "action_name": action_name,
        }

    def _rollout(
        self,
        initial_state: torch.Tensor,
        first_action_oh: torch.Tensor,
        subsequent_action_oh: torch.Tensor,
    ) -> np.ndarray:
        """Roll out N simulations from initial state.

        First step uses first_action_oh, subsequent steps use subsequent_action_oh.
        Adds small noise for stochastic diversity.

        Returns:
            rewards: (n_simulations, horizon) array.
        """
        rewards = np.zeros((self.n_simulations, self.horizon), dtype=np.float32)

        # Batch all simulations
        state_batch = initial_state.unsqueeze(0).expand(self.n_simulations, -1).clone()

        for t in range(self.horizon):
            # Add small noise for stochastic diversity
            noise = torch.randn_like(state_batch) * 0.01
            noisy_state = state_batch + noise

            action = first_action_oh if t == 0 else subsequent_action_oh
            action_batch = action.unsqueeze(0).expand(self.n_simulations, -1)

            next_state, reward, done_prob = self.model(noisy_state, action_batch)

            rewards[:, t] = reward.squeeze(-1).cpu().numpy()
            state_batch = next_state

            # Optionally terminate early if done probability is high
            # (but continue for comparison purposes)

        return rewards
