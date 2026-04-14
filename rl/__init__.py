"""
SupplyMind RL — Gymnasium-compatible RL environments for supply chain risk management.

Registers three difficulty tiers:
  - SupplyMind-Easy-v1   (Typhoon Response, 30 steps)
  - SupplyMind-Medium-v1 (Multi-Front Crisis, 45 steps)
  - SupplyMind-Hard-v1   (Cascading Crisis, 60 steps)
"""

from gymnasium.envs.registration import register

register(
    id="SupplyMind-Easy-v1",
    entry_point="rl.gym_env:SupplyMindGymnasiumEnv",
    kwargs={"task_id": "easy_typhoon_response"},
    max_episode_steps=30,
)

register(
    id="SupplyMind-Medium-v1",
    entry_point="rl.gym_env:SupplyMindGymnasiumEnv",
    kwargs={"task_id": "medium_multi_front"},
    max_episode_steps=45,
)

register(
    id="SupplyMind-Hard-v1",
    entry_point="rl.gym_env:SupplyMindGymnasiumEnv",
    kwargs={"task_id": "hard_cascading_crisis"},
    max_episode_steps=60,
)
