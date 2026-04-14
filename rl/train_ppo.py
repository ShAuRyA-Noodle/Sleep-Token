"""
MaskablePPO training for SupplyMind environments.

Uses sb3-contrib MaskablePPO with:
  - 32 parallel envs via DummyVecEnv (Windows-safe)
  - VecNormalize(norm_obs=True, norm_reward=True)
  - n_steps=2048, batch_size=512, lr=3e-4, device="cuda"
  - 2M total timesteps (~8 min on RTX 4080)
  - MLflow + W&B logging

Usage:
    python -m rl.train_ppo --task easy --steps 2000000
    python -m rl.train_ppo --task medium --steps 2000000
    python -m rl.train_ppo --task hard --steps 2000000
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU optimizations (REQUIRED per kickoff spec)
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASK_MAP = {
    "easy": "SupplyMind-Easy-v1",
    "medium": "SupplyMind-Medium-v1",
    "hard": "SupplyMind-Hard-v1",
}

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _get_action_masks(env) -> np.ndarray:
    """Extract action masks for MultiDiscrete([7, 40]).

    MaskablePPO with MultiDiscrete needs a flat array of length sum(nvec) = 47,
    where first 7 bools mask action types, next 40 mask node indices.
    """
    unwrapped = env.unwrapped
    full_mask = unwrapped._compute_action_mask()  # (280,) = 7*40

    # Reduce: action_type i is valid if ANY node is valid for it
    type_mask = np.zeros(7, dtype=np.bool_)
    for i in range(7):
        type_mask[i] = full_mask[i * 40:(i + 1) * 40].any()

    # Reduce: node j is valid if ANY action type is valid for it
    node_mask = np.zeros(40, dtype=np.bool_)
    for j in range(40):
        node_mask[j] = full_mask[j::40].any()

    return np.concatenate([type_mask, node_mask])  # (47,)


def make_env(env_id: str, seed: int, rank: int) -> Callable:
    """Create a thunk that returns a Gymnasium env with ActionMasker."""
    def _init():
        import gymnasium as gym
        import rl  # noqa: F401 — triggers registration
        from sb3_contrib.common.wrappers import ActionMasker
        from rl.gym_env import SupplyMindGymnasiumEnv
        env = SupplyMindGymnasiumEnv(task_id=env_id.replace("SupplyMind-Easy-v1", "easy_typhoon_response").replace("SupplyMind-Medium-v1", "medium_multi_front").replace("SupplyMind-Hard-v1", "hard_cascading_crisis"), training_mode=True)
        env.reset(seed=seed + rank)
        env = ActionMasker(env, _get_action_masks)
        return env
    return _init


def train_ppo(
    task: str = "easy",
    total_timesteps: int = 2_000_000,
    n_envs: int = 32,
    seed: int = 42,
    device: str = "cuda",
    log_wandb: bool = False,
    log_mlflow: bool = False,
) -> Path:
    """Train MaskablePPO agent and save best model."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_id = TASK_MAP[task]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MaskablePPO Training")
    logger.info("  Task: %s (%s)", task, env_id)
    logger.info("  Timesteps: %s", f"{total_timesteps:,}")
    logger.info("  Parallel envs: %d (DummyVecEnv — Windows-safe)", n_envs)
    logger.info("  Device: %s", device)
    logger.info("  GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("=" * 60)

    # --- W&B ---
    if log_wandb:
        try:
            import wandb
            wandb.init(
                project="supplymind-grand-finale",
                config={
                    "algorithm": "MaskablePPO",
                    "task": task,
                    "total_timesteps": total_timesteps,
                    "n_envs": n_envs,
                    "n_steps": 2048,
                    "batch_size": 512,
                    "learning_rate": 3e-4,
                    "device": device,
                },
                tags=["ppo", task, "phase2"],
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
            log_wandb = False

    # --- MLflow ---
    if log_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("supplymind-ppo")
            mlflow.start_run(run_name=f"ppo-{task}-{seed}")
            mlflow.log_params({
                "algorithm": "MaskablePPO",
                "task": task,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "n_steps": 2048,
                "batch_size": 512,
                "learning_rate": 3e-4,
            })
        except Exception as e:
            logger.warning("MLflow init failed: %s", e)
            log_mlflow = False

    # --- Vectorized environment (DummyVecEnv for Windows) ---
    train_env = DummyVecEnv([make_env(env_id, seed, i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env(env_id, seed + 1000, 0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # --- Custom callback for logging ---
    class LoggingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last_log = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_log >= 10_000:
                self._last_log = self.num_timesteps
                if self.logger is not None:
                    ep_info = self.locals.get("infos", [])
                    rewards = [info.get("episode", {}).get("r", None) for info in ep_info if "episode" in info]
                    if rewards:
                        mean_r = np.mean(rewards)
                        logger.info(
                            "[%dk steps] mean_episode_reward=%.4f",
                            self.num_timesteps // 1000, mean_r,
                        )
                        if log_wandb:
                            try:
                                import wandb
                                wandb.log({"mean_reward": mean_r, "step": self.num_timesteps})
                            except Exception:
                                pass
                        if log_mlflow:
                            try:
                                import mlflow
                                mlflow.log_metrics({"mean_reward": mean_r}, step=self.num_timesteps)
                            except Exception:
                                pass
            return True

    # --- Eval callback (saves best model) ---
    best_model_path = CHECKPOINT_DIR / f"ppo_best_{task}"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        eval_freq=max(total_timesteps // (n_envs * 20), 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # --- Model ---
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    # torch.compile requires Triton (Linux-only) — skip on Windows
    if sys.platform != "win32":
        try:
            model.policy = torch.compile(model.policy, mode="reduce-overhead")
            logger.info("torch.compile applied to policy network")
        except Exception:
            pass

    # --- Train ---
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[LoggingCallback(), eval_callback],
        progress_bar=False,
    )
    elapsed = time.time() - start

    # --- Save final model + vecnormalize stats ---
    final_path = CHECKPOINT_DIR / f"ppo_final_{task}.zip"
    model.save(str(final_path))
    vecnorm_path = CHECKPOINT_DIR / f"ppo_vecnormalize_{task}.pkl"
    train_env.save(str(vecnorm_path))

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Time: %.1f minutes", elapsed / 60)
    logger.info("  Best model: %s", best_model_path)
    logger.info("  Final model: %s", final_path)
    logger.info("  VecNormalize: %s", vecnorm_path)
    logger.info("=" * 60)

    # --- Cleanup ---
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

    train_env.close()
    eval_env.close()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train MaskablePPO on SupplyMind")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    train_ppo(
        task=args.task,
        total_timesteps=args.steps,
        n_envs=args.envs,
        seed=args.seed,
        device=args.device,
        log_wandb=args.wandb,
        log_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
