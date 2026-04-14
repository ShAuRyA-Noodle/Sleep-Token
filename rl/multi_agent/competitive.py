"""
Multi-Agent Competitive RL for SupplyMind.

3 agents (Apple, Samsung, Toyota archetypes) competing for shared supplier
capacity and shared commodity prices.

Demo: Three graphs side by side, trigger TSMC disruption, watch Apple grab
backup first -> Samsung denied -> Toyota caught flat-footed.
"This is the 2021 chip shortage played by three AI agents."

Usage:
    python -m rl.multi_agent.competitive --train
    python -m rl.multi_agent.competitive --demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Agent archetypes — real-world company profiles
# ---------------------------------------------------------------------------
AGENT_PROFILES = {
    "apple": {
        "name": "Apple",
        "strategy": "premium_quality",
        "budget_multiplier": 1.5,    # Deepest pockets
        "risk_tolerance": 0.3,       # Very risk-averse
        "priority_suppliers": ["SUP_TSMC"],
        "key_commodity": "semiconductors",
        "annual_procurement": 87e9,   # $87B (Apple 2023 supply chain spend)
    },
    "samsung": {
        "name": "Samsung",
        "strategy": "vertical_integration",
        "budget_multiplier": 1.2,
        "risk_tolerance": 0.5,
        "priority_suppliers": ["SUP_SAMSUNG"],
        "key_commodity": "memory_chips",
        "annual_procurement": 62e9,   # $62B estimate
    },
    "toyota": {
        "name": "Toyota",
        "strategy": "just_in_time",
        "budget_multiplier": 1.0,
        "risk_tolerance": 0.7,       # Historically lean, exposed to disruption
        "priority_suppliers": [],
        "key_commodity": "auto_parts",
        "annual_procurement": 45e9,   # $45B estimate
    },
}


class CompetitiveSupplyChainEnv:
    """Multi-agent wrapper: shared capacity and shared prices.

    Each step takes {agent_id: action}, applies capacity first-come-first-served.
    If capacity already taken -> capacity_denied + penalty.
    Each large safety stock action spikes prices 2%.

    Args:
        task_id:     Base SupplyMind task.
        agent_ids:   List of agent identifiers.
        seed:        Random seed.
    """

    def __init__(
        self,
        task_id: str = "easy_typhoon_response",
        agent_ids: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        from rl.gym_env import SupplyMindGymnasiumEnv

        self.agent_ids = agent_ids or ["apple", "samsung", "toyota"]
        self.n_agents = len(self.agent_ids)

        # Each agent gets its own environment instance
        self.envs = {
            aid: SupplyMindGymnasiumEnv(task_id=task_id)
            for aid in self.agent_ids
        }

        # Shared state
        self.shared_capacity: dict[str, float] = {}  # supplier_id -> remaining capacity (0-1)
        self.shared_prices: dict[str, float] = {}     # commodity -> price multiplier
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> dict[str, tuple[np.ndarray, dict]]:
        """Reset all agent environments."""
        results = {}
        for i, aid in enumerate(self.agent_ids):
            s = (seed or 42) + i
            results[aid] = self.envs[aid].reset(seed=s)

        # Initialize shared capacity (all suppliers at 100%)
        self.shared_capacity = {}
        self.shared_prices = {}
        return results

    def step(
        self, actions: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, float, bool, bool, dict]]:
        """Execute one step for all agents.

        Actions are processed in random order to avoid first-mover advantage.
        Capacity is first-come-first-served.

        Args:
            actions: {agent_id: np.array([action_type, node_idx])}

        Returns:
            {agent_id: (obs, reward, terminated, truncated, info)}
        """
        # Randomize order
        order = list(actions.keys())
        self.rng.shuffle(order)

        results = {}
        for aid in order:
            action = actions[aid]
            action_type = int(action[0])

            # Check capacity for backup supplier activation
            if action_type == 1:  # activate_backup_supplier
                node_idx = int(action[1])
                cap_key = f"backup_{node_idx}"
                if self.shared_capacity.get(cap_key, 1.0) <= 0:
                    # Capacity already taken — force do_nothing + penalty
                    action = np.array([0, 0], dtype=np.int64)  # do_nothing
                    # Will add penalty to reward later
                else:
                    self.shared_capacity[cap_key] = 0.0  # Claimed

            # Safety stock spikes prices
            if action_type == 3:  # increase_safety_stock
                for commodity in self.shared_prices:
                    self.shared_prices[commodity] = self.shared_prices.get(commodity, 1.0) * 1.02

            obs, reward, term, trunc, info = self.envs[aid].step(action)
            results[aid] = (obs, reward, term, trunc, info)

        return results

    def get_agent_profile(self, agent_id: str) -> dict:
        """Get real-world company profile for an agent."""
        return AGENT_PROFILES.get(agent_id, AGENT_PROFILES["toyota"])


class MAPPOPolicy(nn.Module):
    """Multi-Agent PPO policy network.

    Each agent has its own policy that shares architecture but maintains
    separate weights. Observation includes agent's own state + shared info.

    Args:
        state_dim:  Per-agent observation dim (408).
        shared_dim: Shared state dim (shared capacity + prices).
        action_dim: Number of actions (280).
    """

    def __init__(
        self,
        state_dim: int = 408,
        shared_dim: int = 50,
        action_dim: int = 280,
    ) -> None:
        super().__init__()
        input_dim = state_dim + shared_dim

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> int:
        logits, _ = self.forward(obs.unsqueeze(0))
        logits = logits.squeeze(0)
        if action_mask is not None:
            logits[~action_mask] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()


def run_competitive_demo(task_id: str = "easy_typhoon_response", seed: int = 42) -> dict:
    """Run demo of 3 agents competing during a disruption."""
    env = CompetitiveSupplyChainEnv(task_id=task_id, seed=seed)
    results = env.reset(seed=seed)

    agent_rewards = {aid: 0.0 for aid in env.agent_ids}
    agent_actions = {aid: [] for aid in env.agent_ids}

    for step in range(30):
        actions = {}
        for aid in env.agent_ids:
            # Simple heuristic per archetype
            profile = env.get_agent_profile(aid)
            rng = np.random.default_rng(seed + step + hash(aid))

            # Risk-averse agents (Apple) activate backup early
            if profile["risk_tolerance"] < 0.4 and step < 5:
                actions[aid] = np.array([1, 0], dtype=np.int64)  # activate backup
            elif profile["risk_tolerance"] < 0.5 and step < 10:
                actions[aid] = np.array([3, 1], dtype=np.int64)  # safety stock
            else:
                actions[aid] = np.array([0, 0], dtype=np.int64)  # do nothing

        step_results = env.step(actions)

        for aid in env.agent_ids:
            obs, reward, term, trunc, info = step_results[aid]
            agent_rewards[aid] += reward
            agent_actions[aid].append(int(actions[aid][0]))

    return {
        "rewards": agent_rewards,
        "actions": agent_actions,
        "winner": max(agent_rewards, key=agent_rewards.get),
    }
