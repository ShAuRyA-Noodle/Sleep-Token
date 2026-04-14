"""
SupplyMind AutoResearch — Autonomous RL Experiment System.

Inspired by karpathy/autoresearch. Autonomously runs experiments with
different hyperparameters, reward shaping, and policies to maximize
the grader score. Each experiment has a fixed time/step budget.

Key difference from manual training: this system tries MANY configurations
automatically and keeps the best, like a machine learning researcher
running experiments overnight.

Usage:
    python -m rl.autoresearch --n-experiments 20 --budget 50000
    python -m rl.autoresearch --n-experiments 5 --budget 10000 --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "autoresearch_results"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    # QR-DQN hyperparameters
    lr: float = 3e-4
    n_quantiles: int = 51
    hidden_dim: int = 256
    cvar_alpha: float = 0.1
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 256
    target_update_freq: int = 1000
    # Reward shaping
    grade_reward: bool = True
    alert_penalty: float = 0.01
    real_action_bonus: float = 0.02
    # Training budget
    total_steps: int = 50_000
    # Task
    task_id: str = "easy_typhoon_response"


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    config: ExperimentConfig
    grade_easy: float = 0.0
    grade_medium: float = 0.0
    grade_hard: float = 0.0
    grade_avg: float = 0.0
    reward_easy: float = 0.0
    training_time_min: float = 0.0
    final_loss: float = 0.0
    checkpoint_path: str = ""


def generate_experiment_configs(n: int, base_budget: int = 50_000) -> list[ExperimentConfig]:
    """Generate diverse experiment configurations.

    Varies: learning rate, CVaR alpha, reward shaping, network size.
    """
    rng = np.random.default_rng(42)
    configs = []

    # Always include the baseline
    configs.append(ExperimentConfig(
        name="baseline_grade_reward",
        grade_reward=True,
        total_steps=base_budget,
    ))

    # Systematic variations
    variations = {
        "lr": [1e-4, 3e-4, 1e-3],
        "cvar_alpha": [0.05, 0.1, 0.2, 0.5],
        "hidden_dim": [128, 256, 512],
        "alert_penalty": [0.0, 0.01, 0.05],
        "real_action_bonus": [0.0, 0.02, 0.05, 0.1],
        "gamma": [0.95, 0.99, 0.999],
        "n_quantiles": [21, 51, 101],
    }

    for i in range(1, n):
        # Pick random variations
        config = ExperimentConfig(
            name=f"experiment_{i:03d}",
            lr=float(rng.choice(variations["lr"])),
            cvar_alpha=float(rng.choice(variations["cvar_alpha"])),
            hidden_dim=int(rng.choice(variations["hidden_dim"])),
            alert_penalty=float(rng.choice(variations["alert_penalty"])),
            real_action_bonus=float(rng.choice(variations["real_action_bonus"])),
            gamma=float(rng.choice(variations["gamma"])),
            n_quantiles=int(rng.choice(variations["n_quantiles"])),
            grade_reward=True,
            total_steps=base_budget,
        )
        configs.append(config)

    return configs[:n]


def train_experiment(config: ExperimentConfig, device: str = "cuda") -> ExperimentResult:
    """Run a single experiment: train QR-DQN then evaluate on grader."""
    from rl.distributional.qr_dqn import QRDQNNetwork, quantile_huber_loss
    from rl.gym_env import SupplyMindGymnasiumEnv, ACTION_TYPES, MAX_NODES
    from server.supply_environment import SupplyMindEnvironment
    from collections import deque

    logger.info("  [EXP] %s: lr=%.0e cvar=%.2f hidden=%d grade_rw=%s bonus=%.2f penalty=%.2f",
                config.name, config.lr, config.cvar_alpha, config.hidden_dim,
                config.grade_reward, config.real_action_bonus, config.alert_penalty)

    start = time.time()
    n_actions = 280

    # Environment with grade reward
    env = SupplyMindGymnasiumEnv(
        task_id=config.task_id,
        training_mode=True,
        grade_reward=config.grade_reward,
    )

    # Network
    net = QRDQNNetwork(408, n_actions, config.n_quantiles, config.hidden_dim).to(device)
    target_net = QRDQNNetwork(408, n_actions, config.n_quantiles, config.hidden_dim).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # Simple replay buffer
    buf_states = np.zeros((config.buffer_size, 408), dtype=np.float32)
    buf_actions = np.zeros(config.buffer_size, dtype=np.int64)
    buf_rewards = np.zeros(config.buffer_size, dtype=np.float32)
    buf_next = np.zeros((config.buffer_size, 408), dtype=np.float32)
    buf_dones = np.zeros(config.buffer_size, dtype=np.float32)
    buf_masks = np.zeros((config.buffer_size, n_actions), dtype=np.bool_)
    buf_idx = 0
    buf_size = 0

    obs, info = env.reset(seed=42)
    mask = info["action_masks"]
    recent_rewards = deque(maxlen=50)
    losses = deque(maxlen=100)

    for step in range(1, config.total_steps + 1):
        # Epsilon-greedy with masking
        eps = max(0.05, 1.0 - 0.95 * step / (config.total_steps * 0.5))
        if step < 2000 or np.random.random() < eps:
            valid = np.where(mask)[0]
            flat = np.random.choice(valid) if len(valid) > 0 else 0
        else:
            with torch.no_grad():
                st = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                mk = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
                flat = net.cvar_policy(st, alpha=config.cvar_alpha, action_mask=mk).item()

        action = np.array([flat // 40, flat % 40], dtype=np.int64)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_masks"]

        # Store
        idx = buf_idx % config.buffer_size
        buf_states[idx] = obs
        buf_actions[idx] = flat
        buf_rewards[idx] = reward
        buf_next[idx] = next_obs
        buf_dones[idx] = float(done)
        buf_masks[idx] = mask
        buf_idx += 1
        buf_size = min(buf_size + 1, config.buffer_size)

        if done:
            recent_rewards.append(reward if config.grade_reward else 0)
            obs, info = env.reset()
            mask = info["action_masks"]
        else:
            obs = next_obs
            mask = next_mask

        # Train
        if step >= 2000 and step % 4 == 0:
            idxs = np.random.randint(0, buf_size, config.batch_size)
            s = torch.from_numpy(buf_states[idxs]).to(device)
            a = torch.from_numpy(buf_actions[idxs]).to(device)
            r = torch.from_numpy(buf_rewards[idxs]).to(device)
            ns = torch.from_numpy(buf_next[idxs]).to(device)
            d = torch.from_numpy(buf_dones[idxs]).to(device)

            current_q = net.get_quantile_values(s, a)
            with torch.no_grad():
                next_q_online = net.q_values(ns)
                next_q_online.masked_fill_(~torch.from_numpy(buf_masks[idxs]).to(device), float("-inf"))
                next_a = next_q_online.argmax(dim=-1)
                next_q_target = target_net.get_quantile_values(ns, next_a)
                target_q = r.unsqueeze(1) + config.gamma * (1 - d.unsqueeze(1)) * next_q_target

            loss = quantile_huber_loss(current_q, target_q, net.taus)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())

        if step % config.target_update_freq == 0:
            target_net.load_state_dict(net.state_dict())

    env.close()
    training_time = (time.time() - start) / 60

    # Evaluate on grader (the REAL metric)
    net.eval()
    result = ExperimentResult(config=config, training_time_min=round(training_time, 1))
    result.final_loss = float(np.mean(losses)) if losses else 0.0

    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    task_names = ["easy", "medium", "hard"]

    for task_id, task_name in zip(tasks, task_names):
        grades = []
        rewards = []
        for seed in [42, 99, 7]:
            eval_env = SupplyMindGymnasiumEnv(task_id=task_id)
            eval_core = SupplyMindEnvironment()
            obs_e, info_e = eval_env.reset(seed=seed)
            obs_c = eval_core.reset(task_id=task_id, seed=seed)
            total_r = 0
            while True:
                with torch.no_grad():
                    st = torch.from_numpy(obs_e).float().unsqueeze(0).to(device)
                    mk = torch.from_numpy(info_e["action_masks"]).bool().unsqueeze(0).to(device)
                    flat = net.cvar_policy(st, alpha=config.cvar_alpha, action_mask=mk).item()
                act = np.array([flat // 40, flat % 40], dtype=np.int64)
                obs_e, r, term, trunc, info_e = eval_env.step(act)
                sm = eval_env._decode_action(act)
                obs_c = eval_core.step(sm)
                total_r += r
                if term or trunc or obs_c.done:
                    break
            grades.append(eval_core.grade()["score"])
            rewards.append(total_r)
            eval_env.close()

        grade_avg = float(np.mean(grades))
        if task_name == "easy":
            result.grade_easy = grade_avg
            result.reward_easy = float(np.mean(rewards))
        elif task_name == "medium":
            result.grade_medium = grade_avg
        else:
            result.grade_hard = grade_avg

    result.grade_avg = (result.grade_easy + result.grade_medium + result.grade_hard) / 3

    # Save checkpoint if best
    ckpt_path = CHECKPOINT_DIR / f"autoresearch_{config.name}.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "config": {"state_dim": 408, "n_actions": n_actions,
                   "n_quantiles": config.n_quantiles, "hidden_dim": config.hidden_dim},
        "grade_avg": result.grade_avg,
        "experiment": config.name,
    }, str(ckpt_path))
    result.checkpoint_path = str(ckpt_path)

    logger.info("  [RESULT] %s: easy=%.3f med=%.3f hard=%.3f avg=%.3f (%.1f min)",
                config.name, result.grade_easy, result.grade_medium,
                result.grade_hard, result.grade_avg, training_time)

    del net, target_net
    torch.cuda.empty_cache()
    gc.collect()

    return result


def run_autoresearch(
    n_experiments: int = 20,
    budget_per_experiment: int = 50_000,
    device: str = "cuda",
) -> list[ExperimentResult]:
    """Run autonomous research: many experiments, keep the best."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    configs = generate_experiment_configs(n_experiments, budget_per_experiment)

    logger.info("=" * 70)
    logger.info("SUPPLYMIND AUTORESEARCH")
    logger.info("  Experiments: %d | Budget/exp: %s steps | Device: %s",
                n_experiments, f"{budget_per_experiment:,}", device)
    logger.info("  GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("=" * 70)

    results = []
    best_avg = 0.0
    best_name = ""
    start = time.time()

    for i, config in enumerate(configs):
        logger.info("[%d/%d] Running experiment: %s", i + 1, n_experiments, config.name)
        try:
            result = train_experiment(config, device)
            results.append(result)

            if result.grade_avg > best_avg:
                best_avg = result.grade_avg
                best_name = config.name
                # Copy best checkpoint
                import shutil
                src = Path(result.checkpoint_path)
                dst = CHECKPOINT_DIR / "autoresearch_best.pt"
                shutil.copy2(str(src), str(dst))
                logger.info("  *** NEW BEST: %s (avg=%.3f) ***", best_name, best_avg)
        except Exception as e:
            logger.error("  Experiment %s FAILED: %s", config.name, str(e)[:100])

    total_time = (time.time() - start) / 60

    # Save all results
    results_data = []
    for r in results:
        results_data.append({
            "name": r.config.name,
            "lr": r.config.lr,
            "cvar_alpha": r.config.cvar_alpha,
            "hidden_dim": r.config.hidden_dim,
            "grade_reward": r.config.grade_reward,
            "alert_penalty": r.config.alert_penalty,
            "real_action_bonus": r.config.real_action_bonus,
            "gamma": r.config.gamma,
            "n_quantiles": r.config.n_quantiles,
            "grade_easy": r.grade_easy,
            "grade_medium": r.grade_medium,
            "grade_hard": r.grade_hard,
            "grade_avg": r.grade_avg,
            "reward_easy": r.reward_easy,
            "training_time_min": r.training_time_min,
            "final_loss": r.final_loss,
        })

    output_path = RESULTS_DIR / "autoresearch_results.json"
    output_path.write_text(json.dumps(results_data, indent=2))

    # Leaderboard
    results.sort(key=lambda r: r.grade_avg, reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("AUTORESEARCH COMPLETE — %d experiments in %.1f min", n_experiments, total_time)
    logger.info("=" * 70)
    logger.info("")
    logger.info("LEADERBOARD (by grade avg):")
    logger.info("%-25s  Easy   Med    Hard   Avg    Time", "Experiment")
    logger.info("-" * 70)
    for r in results[:10]:
        logger.info("%-25s  %.3f  %.3f  %.3f  %.3f  %.1fm",
                    r.config.name, r.grade_easy, r.grade_medium,
                    r.grade_hard, r.grade_avg, r.training_time_min)
    logger.info("")
    logger.info("BEST: %s (avg grade = %.3f)", best_name, best_avg)
    logger.info("Best checkpoint: %s", CHECKPOINT_DIR / "autoresearch_best.pt")
    logger.info("=" * 70)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SupplyMind AutoResearch")
    parser.add_argument("--n-experiments", type=int, default=20)
    parser.add_argument("--budget", type=int, default=50_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 experiments, 10K steps")
    args = parser.parse_args()

    if args.quick:
        run_autoresearch(n_experiments=5, budget_per_experiment=10_000, device=args.device)
    else:
        run_autoresearch(n_experiments=args.n_experiments,
                        budget_per_experiment=args.budget, device=args.device)


if __name__ == "__main__":
    main()
