"""
Decision Transformer training for SupplyMind.

Training:
  - Cross-entropy loss on action predictions
  - 10 epochs on 150K transitions from offline buffer
  - ~25 min on RTX 4080 GPU
  - Mixed precision (autocast + GradScaler)

Usage:
    python -m rl.decision_transformer.train
    python -m rl.decision_transformer.train --epochs 20 --context-len 30
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
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl.decision_transformer.model import DecisionTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU optimizations
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SupplyMindTrajectoryDataset(Dataset):
    """Dataset of fixed-length trajectory windows from offline buffer.

    Each sample is a context_len window of (returns_to_go, states, actions, timesteps).
    Sequences shorter than context_len are LEFT-PADDED with zeros and
    attention_mask=0 for padding positions.
    """

    def __init__(
        self,
        npz_path: Path,
        context_len: int = 20,
        max_transitions: int = 150_000,
        n_actions: int = 280,
    ) -> None:
        data = np.load(str(npz_path))
        n = min(len(data["states"]), max_transitions)

        self.states = data["states"][:n].astype(np.float32)
        self.actions_raw = data["actions"][:n].astype(np.int64)
        self.rewards = data["rewards"][:n].astype(np.float32)
        self.dones = data["dones"][:n].astype(np.bool_)
        self.returns_to_go = data["returns_to_go"][:n].astype(np.float32)

        self.context_len = context_len
        self.n_actions = n_actions

        # Build episode boundaries
        self.episode_starts: list[int] = [0]
        for i in range(n):
            if self.dones[i]:
                if i + 1 < n:
                    self.episode_starts.append(i + 1)

        # Build valid sample indices: each (episode_idx, step_within_episode)
        self.samples: list[tuple[int, int]] = []
        for ep_idx in range(len(self.episode_starts)):
            ep_start = self.episode_starts[ep_idx]
            ep_end = self.episode_starts[ep_idx + 1] if ep_idx + 1 < len(self.episode_starts) else n
            ep_len = ep_end - ep_start
            for t in range(ep_len):
                self.samples.append((ep_start, t, ep_len))

        logger.info("Loaded %d transitions, %d episodes, %d samples",
                     n, len(self.episode_starts), len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_start, step, ep_len = self.samples[idx]

        # Extract context window ending at this step
        ctx_start = max(0, step - self.context_len + 1)
        ctx_end = step + 1
        actual_len = ctx_end - ctx_start
        pad_len = self.context_len - actual_len

        # States
        states = np.zeros((self.context_len, self.states.shape[1]), dtype=np.float32)
        states[pad_len:] = self.states[ep_start + ctx_start: ep_start + ctx_end]

        # Returns-to-go
        rtg = np.zeros((self.context_len, 1), dtype=np.float32)
        rtg[pad_len:, 0] = self.returns_to_go[ep_start + ctx_start: ep_start + ctx_end]

        # Actions (one-hot)
        actions = np.zeros((self.context_len, self.n_actions), dtype=np.float32)
        raw_acts = self.actions_raw[ep_start + ctx_start: ep_start + ctx_end]
        for i, act in enumerate(raw_acts):
            flat_idx = int(act[0]) * 40 + int(act[1])
            flat_idx = min(flat_idx, self.n_actions - 1)
            actions[pad_len + i, flat_idx] = 1.0

        # Target: flat action index at current step
        target_act = self.actions_raw[ep_start + step]
        target_flat = int(target_act[0]) * 40 + int(target_act[1])
        target_flat = min(target_flat, self.n_actions - 1)

        # Timesteps
        timesteps = np.zeros(self.context_len, dtype=np.int64)
        timesteps[pad_len:] = np.arange(ctx_start, ctx_end)

        # Attention mask
        attention_mask = np.zeros(self.context_len, dtype=np.float32)
        attention_mask[pad_len:] = 1.0

        return {
            "returns_to_go": torch.from_numpy(rtg),
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "timesteps": torch.from_numpy(timesteps),
            "attention_mask": torch.from_numpy(attention_mask),
            "target_action": torch.tensor(target_flat, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dt(
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    context_len: int = 20,
    max_transitions: int = 150_000,
    seed: int = 42,
    device: str = "cuda",
    log_wandb: bool = False,
    log_mlflow: bool = False,
) -> Path:
    """Train Decision Transformer on offline dataset."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("Decision Transformer Training")
    logger.info("  Epochs: %d | Batch: %d | Context: %d | LR: %.0e",
                epochs, batch_size, context_len, lr)
    logger.info("  Device: %s | GPU: %s", device,
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("=" * 60)

    # --- W&B ---
    if log_wandb:
        try:
            import wandb
            wandb.init(project="supplymind-grand-finale", config={
                "algorithm": "DecisionTransformer", "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "context_len": context_len,
            }, tags=["dt", "phase2"])
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
            log_wandb = False

    # --- MLflow ---
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("supplymind-dt")
            mlflow.start_run(run_name=f"dt-{seed}")
            mlflow.log_params({"algorithm": "DT", "epochs": epochs, "lr": lr, "context_len": context_len})
        except Exception as e:
            logger.warning("MLflow init failed: %s", e)
            log_mlflow = False

    # --- Dataset ---
    npz_path = DATA_DIR / "offline_buffer.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Offline buffer not found at {npz_path}. "
            "Run `python -m rl.offline.dataset` first."
        )

    dataset = SupplyMindTrajectoryDataset(npz_path, context_len, max_transitions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compat (multiprocessing fork issues)
        pin_memory=True if "cuda" in device else False,
        drop_last=True,
    )

    # --- Model ---
    model = DecisionTransformer(
        state_dim=408,
        action_dim=280,
        n_embd=128,
        n_layer=3,
        n_head=1,
        context_len=context_len,
        max_timestep=60,
        dropout=0.1,
    ).to(device)

    if sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile applied")
        except Exception:
            pass

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # --- Train ---
    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            rtg = batch["returns_to_go"].to(device)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            timesteps = batch["timesteps"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            targets = batch["target_action"].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(rtg, states, actions, timesteps, attn_mask)
                # Take last position logits
                pred_logits = logits[:, -1, :]  # (B, 280)
                loss = criterion(pred_logits, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            preds = pred_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.shape[0]

        scheduler.step()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        accuracy = correct / max(total, 1)
        elapsed = time.time() - start

        logger.info(
            "Epoch %d/%d | loss=%.4f | accuracy=%.4f | lr=%.2e | %.1fs elapsed",
            epoch, epochs, avg_loss, accuracy, scheduler.get_last_lr()[0], elapsed,
        )

        if log_wandb:
            try:
                import wandb
                wandb.log({"loss": avg_loss, "accuracy": accuracy, "epoch": epoch})
            except Exception:
                pass
        if log_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({"loss": avg_loss, "accuracy": accuracy}, step=epoch)
            except Exception:
                pass

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = CHECKPOINT_DIR / "dt_best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "state_dim": 408, "action_dim": 280, "n_embd": 128,
                    "n_layer": 3, "n_head": 1, "context_len": context_len,
                    "max_timestep": 60,
                },
                "epoch": epoch, "best_loss": best_loss, "accuracy": accuracy,
            }, str(save_path))
            logger.info("  → New best model saved (loss=%.4f, acc=%.4f)", best_loss, accuracy)

    total_time = time.time() - start
    logger.info("=" * 60)
    logger.info("DT training complete! Time: %.1f minutes", total_time / 60)
    logger.info("  Best loss: %.4f | Best checkpoint: %s", best_loss, CHECKPOINT_DIR / "dt_best.pt")
    logger.info("=" * 60)

    if log_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    if log_mlflow:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return CHECKPOINT_DIR / "dt_best.pt"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--max-transitions", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    train_dt(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        context_len=args.context_len, max_transitions=args.max_transitions,
        seed=args.seed, device=args.device, log_wandb=args.wandb, log_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
