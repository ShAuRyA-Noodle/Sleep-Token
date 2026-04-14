"""
Hindsight Experience Replay (HER) agent for SupplyMind hard task.

Fixes the sparse reward problem on hard_cascading_crisis by relabeling
failed episodes with achieved goals.

GoalEnv wrapper:
  observation → Dict with observation/achieved_goal/desired_goal
  achieved_goal = [health_score, 1-budget_remaining_ratio, cumulative_loss_rate]
  desired_goal  = [0.8, 0.5, 0.2]

Training:
  SAC + HerReplayBuffer, n_sampled_goal=4, strategy="future", 500K steps
  Expected improvement: 30-50% on hard task sparse-reward episodes.

Usage:
    python -m rl.her_agent --task hard --steps 500000
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class SupplyMindGoalEnv(gym.Env):
    """Goal-conditioned wrapper for HER compatibility.

    Wraps the SupplyMind Gymnasium env to provide Dict observation space:
      - 'observation': 408-float state vector
      - 'achieved_goal': [health_score_norm, budget_used_ratio, loss_rate]
      - 'desired_goal': [0.8, 0.5, 0.2] (target performance)

    Reward is sparse: -1 if goal not achieved, 0 if achieved.
    HER relabels failed episodes with the goals that were actually achieved.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Desired goals: health >= 80%, budget used <= 50%, loss rate <= 20%
    DESIRED_GOAL = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    GOAL_TOLERANCE = np.array([0.1, 0.15, 0.1], dtype=np.float32)

    def __init__(self, task_id: str = "hard_cascading_crisis", render_mode: Optional[str] = None) -> None:
        super().__init__()
        import rl  # noqa: F401
        from rl.gym_env import SupplyMindGymnasiumEnv

        self._inner = SupplyMindGymnasiumEnv(task_id=task_id, render_mode=render_mode)
        self.render_mode = render_mode

        obs_space = self._inner.observation_space
        goal_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })
        self.action_space = self._inner.action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        obs, info = self._inner.reset(seed=seed, options=options)
        achieved = self._extract_goal(info)
        goal_obs = {
            "observation": obs,
            "achieved_goal": achieved,
            "desired_goal": self.DESIRED_GOAL.copy(),
        }
        info["action_masks"] = info.get("action_masks", np.ones(280, dtype=np.bool_))
        return goal_obs, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self._inner.step(action)
        achieved = self._extract_goal(info)
        goal_obs = {
            "observation": obs,
            "achieved_goal": achieved,
            "desired_goal": self.DESIRED_GOAL.copy(),
        }
        reward = self.compute_reward(achieved, self.DESIRED_GOAL, info)
        return goal_obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | None = None,
    ) -> float:
        """Sparse reward: 0 if goal achieved within tolerance, -1 otherwise.

        Required by HER — must work with batched goals too.
        """
        if achieved_goal.ndim == 1:
            # Single sample
            diff = np.abs(achieved_goal - desired_goal)
            # Check: health >= desired (higher is better)
            # budget_used <= desired (lower is better)
            # loss_rate <= desired (lower is better)
            health_ok = achieved_goal[0] >= desired_goal[0] - self.GOAL_TOLERANCE[0]
            budget_ok = achieved_goal[1] <= desired_goal[1] + self.GOAL_TOLERANCE[1]
            loss_ok = achieved_goal[2] <= desired_goal[2] + self.GOAL_TOLERANCE[2]
            return 0.0 if (health_ok and budget_ok and loss_ok) else -1.0
        else:
            # Batched
            health_ok = achieved_goal[:, 0] >= desired_goal[:, 0] - self.GOAL_TOLERANCE[0]
            budget_ok = achieved_goal[:, 1] <= desired_goal[:, 1] + self.GOAL_TOLERANCE[1]
            loss_ok = achieved_goal[:, 2] <= desired_goal[:, 2] + self.GOAL_TOLERANCE[2]
            return np.where(health_ok & budget_ok & loss_ok, 0.0, -1.0)

    def _extract_goal(self, info: dict) -> np.ndarray:
        """Extract achieved goal from environment info/observation."""
        raw_obs = info.get("raw_obs", None)
        if raw_obs is None:
            return np.zeros(3, dtype=np.float32)

        fin = raw_obs.financials
        health = fin.supply_chain_health_score / 100.0
        budget_used = 1.0 - (fin.budget_remaining / max(fin.budget_total, 1.0))
        loss_rate = fin.cumulative_revenue_lost / max(fin.total_revenue_at_risk, 1.0)

        return np.array([
            np.clip(health, 0, 1),
            np.clip(budget_used, 0, 1),
            np.clip(loss_rate, 0, 1),
        ], dtype=np.float32)

    def render(self):
        return self._inner.render()


def train_her(
    task: str = "hard",
    total_timesteps: int = 500_000,
    n_sampled_goal: int = 4,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    """Train SAC + HER on goal-conditioned supply chain env."""
    from stable_baselines3 import HerReplayBuffer, SAC

    task_map = {"easy": "easy_typhoon_response", "medium": "medium_multi_front", "hard": "hard_cascading_crisis"}
    task_id = task_map[task]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HER Training (Hindsight Experience Replay)")
    logger.info("  Task: %s | Steps: %s | n_sampled_goal: %d",
                task, f"{total_timesteps:,}", n_sampled_goal)
    logger.info("  Strategy: future | Desired goal: [health>=0.8, budget<=0.5, loss<=0.2]")
    logger.info("  Device: %s", device)
    logger.info("=" * 60)

    env = SupplyMindGoalEnv(task_id=task_id)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            net_arch=[256, 128],
        ),
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    elapsed = time.time() - start

    save_path = CHECKPOINT_DIR / f"her_sac_{task}.zip"
    model.save(str(save_path))

    logger.info("=" * 60)
    logger.info("HER done in %.1f min. Model: %s", elapsed / 60, save_path)
    logger.info("=" * 60)

    env.close()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return save_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train HER agent on SupplyMind")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_her(task=args.task, total_timesteps=args.steps,
              n_sampled_goal=args.n_sampled_goal, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
