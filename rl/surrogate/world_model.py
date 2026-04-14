"""
Neural surrogate world model for SupplyMind.

Learns (state, action) -> (next_state, reward, done) from offline data.
Enables:
  1. GPU Monte Carlo: 100K scenarios in <80ms (vs seconds in Python)
  2. Counterfactual analysis: "Without this action, additional loss = $X"

Architecture:
    Linear(408+280, 512) -> ReLU -> Linear(512, 256) -> ReLU ->
      state_head(256 -> 408)
      reward_head(256 -> 1)
      done_head(256 -> 1) + Sigmoid

Training: 500K transitions, MSE on state/reward, BCE on done, ~4 min GPU.

Usage:
    python -m rl.surrogate.world_model --train
    python -m rl.surrogate.world_model --eval
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class WorldModel(nn.Module):
    """Neural surrogate: (state, action_onehot) -> (next_state, reward, done).

    Architecture:
        shared: Linear(688, 512) -> ReLU -> Linear(512, 256) -> ReLU
        state_head:  Linear(256, 408)
        reward_head: Linear(256, 1)
        done_head:   Linear(256, 1) + Sigmoid
    """

    def __init__(self, state_dim: int = 408, action_dim: int = 280) -> None:
        super().__init__()
        input_dim = state_dim + action_dim  # 688

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.state_head = nn.Linear(256, state_dim)
        self.reward_head = nn.Linear(256, 1)
        self.done_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state:         (batch, 408)
            action_onehot: (batch, 280) one-hot encoded action.

        Returns:
            next_state: (batch, 408) predicted next state.
            reward:     (batch, 1) predicted reward.
            done_prob:  (batch, 1) probability episode ends.
        """
        x = torch.cat([state, action_onehot], dim=-1)
        features = self.shared(x)
        next_state = self.state_head(features)
        reward = self.reward_head(features)
        done_prob = self.done_head(features)
        return next_state, reward, done_prob

    @torch.no_grad()
    def predict(
        self, state: np.ndarray, action_flat: int,
    ) -> tuple[np.ndarray, float, float]:
        """Single-sample prediction (numpy in/out).

        Args:
            state:       (408,) state vector.
            action_flat: int action index (0-279).

        Returns:
            next_state:  (408,) predicted next state.
            reward:      float predicted reward.
            done_prob:   float probability of episode end.
        """
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_t = torch.zeros(1, 280, device=device)
        action_t[0, min(action_flat, 279)] = 1.0

        ns, r, d = self.forward(state_t, action_t)
        return ns[0].cpu().numpy(), r[0, 0].item(), d[0, 0].item()


def train_world_model(
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 3e-4,
    max_transitions: int = 500_000,
    device: str = "cuda",
    seed: int = 42,
) -> Path:
    """Train world model on offline dataset. ~4 min on RTX 4080."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("World Model Training")
    logger.info("  Epochs: %d | Batch: %d | LR: %.0e | Device: %s", epochs, batch_size, lr, device)
    logger.info("=" * 60)

    # Load data
    npz_path = DATA_DIR / "offline_buffer.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Offline buffer not found at {npz_path}")

    data = np.load(str(npz_path))
    n = min(len(data["states"]), max_transitions)
    states = data["states"][:n]
    actions = data["actions"][:n]
    rewards = data["rewards"][:n]
    next_states = data["next_states"][:n]
    dones = data["dones"][:n].astype(np.float32)

    # Convert actions to one-hot
    action_onehot = np.zeros((n, 280), dtype=np.float32)
    for i in range(n):
        flat = int(actions[i, 0]) * 40 + int(actions[i, 1])
        action_onehot[i, min(flat, 279)] = 1.0

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(action_onehot),
        torch.from_numpy(next_states),
        torch.from_numpy(rewards).unsqueeze(1),
        torch.from_numpy(dones).unsqueeze(1),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model = WorldModel(408, 280).to(device)
    if sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_state_loss = 0.0
        total_reward_loss = 0.0
        total_done_loss = 0.0
        batches = 0

        for s, a, ns, r, d in loader:
            s, a, ns, r, d = s.to(device), a.to(device), ns.to(device), r.to(device), d.to(device)

            with torch.amp.autocast("cuda"):
                pred_ns, pred_r, pred_d = model(s, a)
                loss_state = mse(pred_ns, ns)
                loss_reward = mse(pred_r, r)
            # BCE unsafe under autocast — compute in fp32 outside
            loss_done = bce(pred_d.float(), d.float())
            loss = loss_state + loss_reward + 0.1 * loss_done

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_state_loss += loss_state.item()
            total_reward_loss += loss_reward.item()
            total_done_loss += loss_done.item()
            batches += 1

        avg_sl = total_state_loss / batches
        avg_rl = total_reward_loss / batches
        avg_dl = total_done_loss / batches
        total = avg_sl + avg_rl + 0.1 * avg_dl

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  Epoch %d/%d | state=%.6f | reward=%.6f | done=%.6f | total=%.6f",
                epoch, epochs, avg_sl, avg_rl, avg_dl, total,
            )

        if total < best_loss:
            best_loss = total
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"state_dim": 408, "action_dim": 280},
                "epoch": epoch, "loss": best_loss,
            }, str(CHECKPOINT_DIR / "world_model_best.pt"))

    elapsed = time.time() - start
    logger.info("World model training done in %.1f min. Best loss: %.6f", elapsed / 60, best_loss)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return CHECKPOINT_DIR / "world_model_best.pt"


def load_world_model(device: str = "cuda") -> WorldModel:
    """Load trained world model from checkpoint."""
    path = CHECKPOINT_DIR / "world_model_best.pt"
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    model = WorldModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Neural surrogate world model")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.train:
        train_world_model(device=args.device)
    if args.eval:
        model = load_world_model(device=args.device)
        print(f"World model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == "__main__":
    main()
