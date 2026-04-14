"""
Optuna hyperparameter optimization sweep for SupplyMind.

50 trials x 500K steps — run overnight on GPU.

Tunes PPO hyperparameters: learning_rate, n_steps, clip_range,
ent_coef, gamma, gae_lambda, net_arch.

Usage:
    python -m rl.hpo --n-trials 50 --task easy
    python -m rl.hpo --n-trials 10 --task easy --quick   # quick test
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

TASK_MAP = {
    "easy": "SupplyMind-Easy-v1",
    "medium": "SupplyMind-Medium-v1",
    "hard": "SupplyMind-Hard-v1",
}


def objective(trial, task: str = "easy", total_timesteps: int = 500_000) -> float:
    """Optuna objective function for PPO hyperparameter tuning.

    Args:
        trial:           Optuna trial object.
        task:            Task difficulty.
        total_timesteps: Training budget per trial.

    Returns:
        Mean evaluation reward (to maximize).
    """
    import optuna
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.train_ppo import make_env

    env_id = TASK_MAP[task]

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch_map = {
        "small": dict(pi=[128, 64], vf=[128, 64]),
        "medium": dict(pi=[256, 128], vf=[256, 128]),
        "large": dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    }

    # Environment
    n_envs = 8  # Fewer envs per trial to save time
    train_env = DummyVecEnv([make_env(env_id, 42, i) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    try:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0,
            policy_kwargs=dict(
                net_arch=net_arch_map[net_arch_choice],
                activation_fn=torch.nn.ReLU,
            ),
        )

        model.learn(total_timesteps=total_timesteps, progress_bar=False)

        # Evaluate
        eval_env = DummyVecEnv([make_env(env_id, 1000, 0)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        rewards = []
        for _ in range(10):
            obs = eval_env.reset()
            total_r = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, dones, infos = eval_env.step(action)
                total_r += r[0]
                done = dones[0]
            rewards.append(total_r)

        mean_reward = float(np.mean(rewards))

        eval_env.close()
        train_env.close()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return mean_reward

    except Exception as e:
        logger.warning("Trial %d failed: %s", trial.number, e)
        train_env.close()
        torch.cuda.empty_cache()
        gc.collect()
        return float("-inf")


def run_hpo(
    n_trials: int = 50,
    task: str = "easy",
    total_timesteps: int = 500_000,
) -> dict:
    """Run Optuna HPO sweep."""
    import optuna

    logger.info("=" * 60)
    logger.info("OPTUNA HPO SWEEP")
    logger.info("  Trials: %d | Task: %s | Steps/trial: %s",
                n_trials, task, f"{total_timesteps:,}")
    logger.info("=" * 60)

    # In-memory storage (avoids SQLite conflict on Windows)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"supplymind-ppo-{task}",
        storage=None,
    )

    start = time.time()
    study.optimize(
        lambda trial: objective(trial, task, total_timesteps),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("HPO complete in %.1f hours", elapsed / 3600)
    logger.info("  Best trial: %d", study.best_trial.number)
    logger.info("  Best reward: %.4f", study.best_value)
    logger.info("  Best params:")
    for k, v in study.best_params.items():
        logger.info("    %s: %s", k, v)
    logger.info("=" * 60)

    # Save results
    import json
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": n_trials,
        "task": task,
        "elapsed_seconds": elapsed,
    }
    output_path = CHECKPOINT_DIR / "hpo_results.json"
    output_path.write_text(json.dumps(results, indent=2))

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Optuna HPO for SupplyMind PPO")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--quick", action="store_true", help="Quick test (10 trials, 50K steps)")
    args = parser.parse_args()

    if args.quick:
        run_hpo(n_trials=10, task=args.task, total_timesteps=50_000)
    else:
        run_hpo(n_trials=args.n_trials, task=args.task, total_timesteps=args.steps)


if __name__ == "__main__":
    main()
