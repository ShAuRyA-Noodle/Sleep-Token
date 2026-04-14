"""
QR-DQN training loop for SupplyMind environments.

Training details:
  - 200K steps, replay buffer (100K capacity)
  - Target network soft-updated every 1000 steps
  - Epsilon-greedy exploration with action masking
  - Quantile Huber loss (kappa=1.0)
  - ~30 min on RTX 4080

Usage:
    python -m rl.distributional.train --task easy --steps 200000
    python -m rl.distributional.train --task hard --steps 200000 --cvar-alpha 0.05
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl.distributional.qr_dqn import QRDQNNetwork, quantile_huber_loss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU optimizations
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple replay buffer with numpy arrays for speed."""

    def __init__(self, capacity: int, state_dim: int, device: str = "cuda") -> None:
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.action_masks = np.zeros((capacity, 280), dtype=np.bool_)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.action_masks[self.idx] = action_mask
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.from_numpy(self.states[idxs]).to(self.device),
            "actions": torch.from_numpy(self.actions[idxs]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[idxs]).to(self.device),
            "next_states": torch.from_numpy(self.next_states[idxs]).to(self.device),
            "dones": torch.from_numpy(self.dones[idxs].astype(np.float32)).to(self.device),
            "action_masks": torch.from_numpy(self.action_masks[idxs]).to(self.device),
        }


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------

def epsilon_schedule(step: int, total_steps: int) -> float:
    """Linear decay from 1.0 to 0.05 over first 50% of training."""
    warmup_end = total_steps * 0.5
    if step >= warmup_end:
        return 0.05
    return 1.0 - 0.95 * (step / warmup_end)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_qrdqn(
    task: str = "easy",
    total_steps: int = 200_000,
    buffer_capacity: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    target_update_freq: int = 1000,
    train_freq: int = 4,
    learning_starts: int = 5000,
    n_quantiles: int = 51,
    cvar_alpha: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    log_wandb: bool = False,
    log_mlflow: bool = False,
) -> Path:
    """Train QR-DQN agent with action masking."""
    import gymnasium as gym
    import rl  # noqa: F401

    task_map = {
        "easy": "SupplyMind-Easy-v1",
        "medium": "SupplyMind-Medium-v1",
        "hard": "SupplyMind-Hard-v1",
    }
    env_id = task_map[task]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("QR-DQN Training")
    logger.info("  Task: %s | Steps: %s | Quantiles: %d | CVaR alpha: %.2f",
                task, f"{total_steps:,}", n_quantiles, cvar_alpha)
    logger.info("  Device: %s | GPU: %s", device,
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("=" * 60)

    # --- W&B ---
    if log_wandb:
        try:
            import wandb
            wandb.init(project="supplymind-grand-finale", config={
                "algorithm": "QR-DQN", "task": task, "n_quantiles": n_quantiles,
                "cvar_alpha": cvar_alpha, "learning_rate": lr, "total_steps": total_steps,
            }, tags=["qrdqn", task, "phase2"])
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
            log_wandb = False

    # --- MLflow ---
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("supplymind-qrdqn")
            mlflow.start_run(run_name=f"qrdqn-{task}-{seed}")
            mlflow.log_params({
                "algorithm": "QR-DQN", "task": task, "n_quantiles": n_quantiles,
                "cvar_alpha": cvar_alpha, "lr": lr, "total_steps": total_steps,
            })
        except Exception as e:
            logger.warning("MLflow init failed: %s", e)
            log_mlflow = False

    # --- Environment (training_mode=True for fast MC) ---
    from rl.gym_env import SupplyMindGymnasiumEnv
    task_id_str = {"SupplyMind-Easy-v1": "easy_typhoon_response", "SupplyMind-Medium-v1": "medium_multi_front", "SupplyMind-Hard-v1": "hard_cascading_crisis"}[env_id]
    env = SupplyMindGymnasiumEnv(task_id=task_id_str, training_mode=True)
    state_dim = env.observation_space.shape[0]  # 408
    n_actions = 280  # 7 types × 40 nodes

    # --- Networks ---
    online_net = QRDQNNetwork(state_dim, n_actions, n_quantiles).to(device)
    target_net = QRDQNNetwork(state_dim, n_actions, n_quantiles).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    # torch.compile requires Triton (Linux-only)
    if sys.platform != "win32":
        try:
            online_net = torch.compile(online_net, mode="reduce-overhead")
            logger.info("torch.compile applied to online network")
        except Exception:
            pass

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    # --- Replay buffer ---
    replay = ReplayBuffer(buffer_capacity, state_dim, device)

    # --- Training loop ---
    obs, info = env.reset(seed=seed)
    action_mask = info["action_masks"]

    best_avg_reward = float("-inf")
    episode_rewards: list[float] = []
    ep_reward = 0.0
    recent_rewards = deque(maxlen=50)
    losses = deque(maxlen=100)
    start_time = time.time()

    for step in range(1, total_steps + 1):
        eps = epsilon_schedule(step, total_steps)

        # --- Action selection with masking ---
        if step < learning_starts or np.random.random() < eps:
            # Random masked action
            valid = np.where(action_mask)[0]
            flat_action = np.random.choice(valid) if len(valid) > 0 else 0
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(device)
                flat_action = online_net.cvar_policy(state_t, alpha=cvar_alpha, action_mask=mask_t).item()

        # Convert flat action to MultiDiscrete
        action_type = flat_action // 40
        node_idx = flat_action % 40
        gym_action = np.array([action_type, node_idx], dtype=np.int64)

        # --- Step ---
        next_obs, reward, terminated, truncated, next_info = env.step(gym_action)
        done = terminated or truncated
        next_mask = next_info["action_masks"]

        replay.add(obs, flat_action, reward, next_obs, done, action_mask)
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)
            recent_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, info = env.reset()
            action_mask = info["action_masks"]
        else:
            obs = next_obs
            action_mask = next_mask

        # --- Train ---
        if step >= learning_starts and step % train_freq == 0:
            batch = replay.sample(batch_size)

            with torch.amp.autocast("cuda"):
                # Current quantile values for selected actions
                current_quantiles = online_net.get_quantile_values(
                    batch["states"], batch["actions"],
                )  # (B, N_quantiles)

                # Target quantile values (double DQN style)
                with torch.no_grad():
                    # Select best action using online net
                    next_q = online_net.q_values(batch["next_states"])
                    next_q.masked_fill_(~batch["action_masks"], float("-inf"))
                    next_actions = next_q.argmax(dim=-1)

                    # Get quantiles from target net for those actions
                    next_quantiles = target_net.get_quantile_values(
                        batch["next_states"], next_actions,
                    )

                    # Bellman target
                    target_quantiles = (
                        batch["rewards"].unsqueeze(1)
                        + gamma * (1 - batch["dones"].unsqueeze(1)) * next_quantiles
                    )

                loss = quantile_huber_loss(
                    current_quantiles, target_quantiles, online_net.taus,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        # --- Target network update ---
        if step % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        # --- Logging ---
        if step % 10_000 == 0:
            elapsed = time.time() - start_time
            avg_r = np.mean(recent_rewards) if recent_rewards else 0
            avg_loss = np.mean(losses) if losses else 0
            logger.info(
                "[%dk/%dk] eps=%.3f | avg_reward=%.4f | loss=%.4f | %.0f steps/s",
                step // 1000, total_steps // 1000, eps, avg_r, avg_loss,
                step / elapsed,
            )

            if log_wandb:
                try:
                    import wandb
                    wandb.log({"mean_reward": avg_r, "loss": avg_loss, "epsilon": eps, "step": step})
                except Exception:
                    pass
            if log_mlflow:
                try:
                    import mlflow
                    mlflow.log_metrics({"mean_reward": avg_r, "loss": avg_loss, "epsilon": eps}, step=step)
                except Exception:
                    pass

            # Save best
            if avg_r > best_avg_reward and len(recent_rewards) >= 10:
                best_avg_reward = avg_r
                save_path = CHECKPOINT_DIR / f"qrdqn_best_{task}.pt"
                torch.save({
                    "state_dict": online_net.state_dict(),
                    "config": {
                        "state_dim": state_dim, "n_actions": n_actions,
                        "n_quantiles": n_quantiles, "cvar_alpha": cvar_alpha,
                    },
                    "step": step, "best_reward": best_avg_reward,
                }, str(save_path))
                logger.info("  → New best model saved (reward=%.4f)", best_avg_reward)

    # --- Save final ---
    final_path = CHECKPOINT_DIR / f"qrdqn_final_{task}.pt"
    torch.save({
        "state_dict": online_net.state_dict(),
        "config": {
            "state_dim": state_dim, "n_actions": n_actions,
            "n_quantiles": n_quantiles, "cvar_alpha": cvar_alpha,
        },
        "step": total_steps,
        "episodes": len(episode_rewards),
        "final_avg_reward": np.mean(recent_rewards) if recent_rewards else 0,
    }, str(final_path))

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("QR-DQN training complete!")
    logger.info("  Time: %.1f minutes", elapsed / 60)
    logger.info("  Episodes: %d | Final avg reward: %.4f",
                len(episode_rewards), np.mean(recent_rewards) if recent_rewards else 0)
    logger.info("  Best checkpoint: %s", CHECKPOINT_DIR / f"qrdqn_best_{task}.pt")
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

    env.close()
    del online_net, target_net
    torch.cuda.empty_cache()
    gc.collect()

    return final_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train QR-DQN on SupplyMind")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--cvar-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    train_qrdqn(task=args.task, total_steps=args.steps, cvar_alpha=args.cvar_alpha,
                seed=args.seed, device=args.device, log_wandb=args.wandb, log_mlflow=args.mlflow)


if __name__ == "__main__":
    main()
