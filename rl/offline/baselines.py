"""
Offline RL baselines for SupplyMind — PURE PYTORCH (no d3rlpy dependency).

CQL:    Conservative Q-Learning with OOD action penalty, 100K steps, ~15 min GPU
TD3+BC: TD3 with Behavior Cloning regularization, 100K steps, ~12 min GPU
BC:     3-layer MLP (408→256→128→280), cross-entropy, ~5 min GPU

Usage:
    python -m rl.offline.baselines --algo cql
    python -m rl.offline.baselines --algo td3bc
    python -m rl.offline.baselines --algo bc
    python -m rl.offline.baselines --algo all
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
import torch.nn.functional as F
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


def _load_offline_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load offline buffer. Returns (states, flat_actions, rewards, next_states, dones)."""
    npz_path = DATA_DIR / "offline_buffer.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Offline buffer not found at {npz_path}. Run dataset.py first.")

    data = np.load(str(npz_path))
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    next_states = data["next_states"].astype(np.float32)
    dones = data["dones"].astype(np.float32)

    flat_actions = (actions[:, 0] * 40 + actions[:, 1]).astype(np.int64)
    return states, flat_actions, rewards, next_states, dones


# ---------------------------------------------------------------------------
# Behavior Cloning — The floor baseline
# ---------------------------------------------------------------------------

class BCNetwork(nn.Module):
    """3-layer MLP for Behavior Cloning.
    Architecture: Linear(408→256) → ReLU → Linear(256→128) → ReLU → Linear(128→280)
    """
    def __init__(self, state_dim: int = 408, action_dim: int = 280) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_bc(
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
    scripted_only: bool = True,
) -> Path:
    """Train Behavior Cloning on scripted agent demonstrations."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("Behavior Cloning Training")
    logger.info("  Epochs: %d | Batch: %d | LR: %.0e | Device: %s", epochs, batch_size, lr, device)
    logger.info("=" * 60)

    states, flat_actions, _, _, _ = _load_offline_data()

    if scripted_only:
        half = len(states) // 2
        states = states[:half]
        flat_actions = flat_actions[:half]
        logger.info("Using scripted-only subset: %d transitions", len(states))

    states_t = torch.from_numpy(states)
    actions_t = torch.from_numpy(flat_actions)
    dataset = TensorDataset(states_t, actions_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True if "cuda" in device else False, drop_last=True)

    model = BCNetwork(408, 280).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(batch_states)
                    loss = criterion(logits, batch_actions)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_states)
                loss = criterion(logits, batch_actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=-1) == batch_actions).sum().item()
            total += batch_actions.shape[0]

        avg_loss = epoch_loss / max(len(dataloader), 1)
        accuracy = correct / max(total, 1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Epoch %d/%d | loss=%.4f | accuracy=%.4f", epoch, epochs, avg_loss, accuracy)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                         "loss": best_loss, "accuracy": accuracy},
                       str(CHECKPOINT_DIR / "bc_best.pt"))

    elapsed = time.time() - start
    logger.info("BC done in %.1f min. Best loss: %.4f. Saved: %s",
                elapsed / 60, best_loss, CHECKPOINT_DIR / "bc_best.pt")
    del model; torch.cuda.empty_cache(); gc.collect()
    return CHECKPOINT_DIR / "bc_best.pt"


# ---------------------------------------------------------------------------
# CQL — Conservative Q-Learning (Pure PyTorch)
# ---------------------------------------------------------------------------

class CQLQNetwork(nn.Module):
    """Twin Q-networks for CQL."""
    def __init__(self, state_dim: int = 408, action_dim: int = 280) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)

    def q_min(self, state: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state)
        return torch.min(q1, q2)


def train_cql(
    n_steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    conservative_weight: float = 5.0,
    device: str = "cuda",
    seed: int = 42,
) -> Path:
    """Train CQL (Conservative Q-Learning) from offline data.

    Key idea: penalize Q-values for out-of-distribution actions to prevent
    overestimation of unseen state-action pairs.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("CQL Training (Conservative Q-Learning) — Pure PyTorch")
    logger.info("  Steps: %s | conservative_weight=%.1f | Device: %s",
                f"{n_steps:,}", conservative_weight, device)
    logger.info("=" * 60)

    states, flat_actions, rewards, next_states, dones = _load_offline_data()
    n = len(states)
    logger.info("  Loaded %d transitions", n)

    # Convert to tensors
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(flat_actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    # Networks
    online_q = CQLQNetwork(408, 280).to(device)
    target_q = CQLQNetwork(408, 280).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optimizer = optim.Adam(online_q.parameters(), lr=lr)

    best_loss = float("inf")
    start = time.time()

    for step in range(1, n_steps + 1):
        # Sample batch
        idx = torch.randint(0, n, (batch_size,), device=device)
        s = states_t[idx]
        a = actions_t[idx]
        r = rewards_t[idx]
        ns = next_states_t[idx]
        d = dones_t[idx]

        # Standard Bellman target (Double DQN style)
        with torch.no_grad():
            next_q_min = target_q.q_min(ns)
            next_actions = online_q.q_min(ns).argmax(dim=-1)
            next_q_val = next_q_min[torch.arange(batch_size, device=device), next_actions]
            target = r + gamma * (1 - d) * next_q_val

        # Current Q values
        q1, q2 = online_q(s)
        q1_a = q1[torch.arange(batch_size, device=device), a]
        q2_a = q2[torch.arange(batch_size, device=device), a]

        # Bellman loss
        bellman_loss = F.mse_loss(q1_a, target) + F.mse_loss(q2_a, target)

        # CQL conservative penalty: logsumexp(Q) - Q(s,a) for both critics
        # This penalizes high Q-values for actions not in the dataset
        cql_penalty = (
            (torch.logsumexp(q1, dim=1).mean() - q1_a.mean()) +
            (torch.logsumexp(q2, dim=1).mean() - q2_a.mean())
        )

        loss = bellman_loss + conservative_weight * cql_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_q.parameters(), 1.0)
        optimizer.step()

        # Soft target update
        if step % 2 == 0:
            tau = 0.005
            for p, tp in zip(online_q.parameters(), target_q.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if step % 10_000 == 0:
            logger.info("  [%dk/%dk] bellman=%.4f cql_penalty=%.4f total=%.4f",
                        step // 1000, n_steps // 1000,
                        bellman_loss.item(), cql_penalty.item(), loss.item())

        if step % 10_000 == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({"state_dict": online_q.state_dict(), "step": step,
                         "loss": best_loss, "config": {"state_dim": 408, "action_dim": 280,
                         "conservative_weight": conservative_weight}},
                       str(CHECKPOINT_DIR / "cql_best.pt"))

    elapsed = time.time() - start
    logger.info("CQL done in %.1f min. Best loss: %.4f", elapsed / 60, best_loss)
    del online_q, target_q; torch.cuda.empty_cache(); gc.collect()
    return CHECKPOINT_DIR / "cql_best.pt"


# ---------------------------------------------------------------------------
# TD3+BC — TD3 with Behavior Cloning regularization (Pure PyTorch)
# ---------------------------------------------------------------------------

class TD3Actor(nn.Module):
    """Deterministic actor for TD3+BC. Outputs action logits (discrete)."""
    def __init__(self, state_dim: int = 408, action_dim: int = 280) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TD3Critic(nn.Module):
    """Twin Q-critics for TD3+BC."""
    def __init__(self, state_dim: int = 408, action_dim: int = 280) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)


def train_td3bc(
    n_steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha: float = 2.5,
    policy_delay: int = 2,
    device: str = "cuda",
    seed: int = 42,
) -> Path:
    """Train TD3+BC from offline data.

    Key idea: TD3 critic + actor that is regularized toward the behavioral
    policy (BC term). alpha controls the BC regularization strength.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("TD3+BC Training — Pure PyTorch")
    logger.info("  Steps: %s | alpha=%.1f | Device: %s", f"{n_steps:,}", alpha, device)
    logger.info("=" * 60)

    states, flat_actions, rewards, next_states, dones = _load_offline_data()
    n = len(states)
    logger.info("  Loaded %d transitions", n)

    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(flat_actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    # One-hot encode actions for BC loss
    actions_onehot = F.one_hot(actions_t.long(), 280).float()

    actor = TD3Actor(408, 280).to(device)
    critic = TD3Critic(408, 280).to(device)
    target_actor = TD3Actor(408, 280).to(device)
    target_critic = TD3Critic(408, 280).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    best_loss = float("inf")
    start = time.time()

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        s = states_t[idx]
        a = actions_t[idx]
        r = rewards_t[idx]
        ns = next_states_t[idx]
        d = dones_t[idx]
        a_oh = actions_onehot[idx]

        # --- Critic update ---
        with torch.no_grad():
            next_logits = target_actor(ns)
            next_a = next_logits.argmax(dim=-1)
            tq1, tq2 = target_critic(ns)
            next_q = torch.min(
                tq1[torch.arange(batch_size, device=device), next_a],
                tq2[torch.arange(batch_size, device=device), next_a],
            )
            target_q = r + gamma * (1 - d) * next_q

        q1, q2 = critic(s)
        q1_a = q1[torch.arange(batch_size, device=device), a]
        q2_a = q2[torch.arange(batch_size, device=device), a]
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()

        # --- Actor update (delayed) ---
        if step % policy_delay == 0:
            logits = actor(s)
            probs = F.softmax(logits, dim=-1)
            pred_a = logits.argmax(dim=-1)

            # Q-value for actor's chosen action
            q1_pi, _ = critic(s)
            q_val = q1_pi[torch.arange(batch_size, device=device), pred_a]

            # Normalize Q for alpha scaling
            lam = alpha / (q_val.abs().mean().detach() + 1e-8)

            # BC loss: cross-entropy between actor output and dataset actions
            bc_loss = F.cross_entropy(logits, a)

            # TD3+BC loss: maximize Q + BC regularization
            actor_loss = -lam * q_val.mean() + bc_loss

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optim.step()

            # Soft target update
            tau = 0.005
            for p, tp in zip(actor.parameters(), target_actor.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if step % 10_000 == 0:
            logger.info("  [%dk/%dk] critic=%.4f",
                        step // 1000, n_steps // 1000, critic_loss.item())

        if step % 10_000 == 0 and critic_loss.item() < best_loss:
            best_loss = critic_loss.item()
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                         "step": step, "loss": best_loss},
                       str(CHECKPOINT_DIR / "td3bc_best.pt"))

    elapsed = time.time() - start
    logger.info("TD3+BC done in %.1f min. Best loss: %.4f", elapsed / 60, best_loss)
    del actor, critic, target_actor, target_critic
    torch.cuda.empty_cache(); gc.collect()
    return CHECKPOINT_DIR / "td3bc_best.pt"


# ---------------------------------------------------------------------------
# IQL — Implicit Q-Learning (Pure PyTorch)
# ---------------------------------------------------------------------------

class IQLValueNet(nn.Module):
    """Value network V(s) for IQL."""
    def __init__(self, state_dim: int = 408) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def train_iql(
    n_steps: int = 100_000,
    batch_size: int = 256,
    actor_lr: float = 1e-4,
    critic_lr: float = 3e-4,
    value_lr: float = 3e-4,
    gamma: float = 0.99,
    expectile: float = 0.7,
    weight_temp: float = 3.0,
    max_weight: float = 100.0,
    device: str = "cuda",
    seed: int = 42,
) -> Path:
    """Train IQL (Implicit Q-Learning) from offline data.

    Key idea: learns V(s) via expectile regression on Q(s,a), avoiding
    querying Q for OOD actions entirely. The production-relevant paradigm.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    logger.info("=" * 60)
    logger.info("IQL Training (Implicit Q-Learning) — Pure PyTorch")
    logger.info("  Steps: %s | expectile=%.1f | weight_temp=%.1f | Device: %s",
                f"{n_steps:,}", expectile, weight_temp, device)
    logger.info("=" * 60)

    states, flat_actions, rewards, next_states, dones = _load_offline_data()
    n = len(states)
    logger.info("  Loaded %d transitions", n)

    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(flat_actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    # Networks
    q_net = CQLQNetwork(408, 280).to(device)  # Reuse twin Q architecture
    value_net = IQLValueNet(408).to(device)
    actor = BCNetwork(408, 280).to(device)  # Policy extracts from Q
    target_q = CQLQNetwork(408, 280).to(device)
    target_q.load_state_dict(q_net.state_dict())

    q_optim = optim.Adam(q_net.parameters(), lr=critic_lr)
    v_optim = optim.Adam(value_net.parameters(), lr=value_lr)
    a_optim = optim.Adam(actor.parameters(), lr=actor_lr)

    best_loss = float("inf")
    start = time.time()

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        s = states_t[idx]
        a = actions_t[idx]
        r = rewards_t[idx]
        ns = next_states_t[idx]
        d = dones_t[idx]

        # --- Value function update (expectile regression) ---
        with torch.no_grad():
            tq1, tq2 = target_q(s)
            q_a = torch.min(
                tq1[torch.arange(batch_size, device=device), a],
                tq2[torch.arange(batch_size, device=device), a],
            )

        v = value_net(s)
        diff = q_a - v
        # Expectile loss: asymmetric L2 — heavier weight on positive errors (high Q)
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        value_loss = (weight * diff.pow(2)).mean()

        v_optim.zero_grad()
        value_loss.backward()
        v_optim.step()

        # --- Q-function update (standard Bellman with V as target) ---
        with torch.no_grad():
            next_v = value_net(ns)
            q_target = r + gamma * (1 - d) * next_v

        q1, q2 = q_net(s)
        q1_a = q1[torch.arange(batch_size, device=device), a]
        q2_a = q2[torch.arange(batch_size, device=device), a]
        q_loss = F.mse_loss(q1_a, q_target) + F.mse_loss(q2_a, q_target)

        q_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        q_optim.step()

        # --- Actor update (advantage-weighted regression) ---
        with torch.no_grad():
            v_s = value_net(s)
            tq1, tq2 = target_q(s)
            q_a_for_adv = torch.min(
                tq1[torch.arange(batch_size, device=device), a],
                tq2[torch.arange(batch_size, device=device), a],
            )
            advantage = q_a_for_adv - v_s
            # Exponentiated advantage with temperature and clipping
            exp_adv = torch.exp(advantage * weight_temp).clamp(max=max_weight)

        logits = actor(s)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob_a = log_probs[torch.arange(batch_size, device=device), a]
        actor_loss = -(exp_adv * log_prob_a).mean()

        a_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        a_optim.step()

        # Soft target update
        if step % 2 == 0:
            tau = 0.005
            for p, tp in zip(q_net.parameters(), target_q.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if step % 10_000 == 0:
            logger.info("  [%dk/%dk] q_loss=%.4f v_loss=%.4f actor_loss=%.4f",
                        step // 1000, n_steps // 1000,
                        q_loss.item(), value_loss.item(), actor_loss.item())

        total = q_loss.item() + value_loss.item()
        if step % 10_000 == 0 and total < best_loss:
            best_loss = total
            torch.save({
                "q_net": q_net.state_dict(),
                "value_net": value_net.state_dict(),
                "actor": actor.state_dict(),
                "step": step, "loss": best_loss,
                "config": {"expectile": expectile, "weight_temp": weight_temp},
            }, str(CHECKPOINT_DIR / "iql_best.pt"))

    elapsed = time.time() - start
    logger.info("IQL done in %.1f min. Best loss: %.4f", elapsed / 60, best_loss)
    del q_net, value_net, actor, target_q
    torch.cuda.empty_cache(); gc.collect()
    return CHECKPOINT_DIR / "iql_best.pt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train offline RL baselines (Pure PyTorch)")
    parser.add_argument("--algo", choices=["cql", "td3bc", "bc", "iql", "all"], default="all")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.algo in ("bc", "all"):
        train_bc(device=args.device, seed=args.seed)
    if args.algo in ("cql", "all"):
        train_cql(n_steps=args.steps, device=args.device, seed=args.seed)
    if args.algo in ("td3bc", "all"):
        train_td3bc(n_steps=args.steps, device=args.device, seed=args.seed)
    if args.algo in ("iql", "all"):
        train_iql(n_steps=args.steps, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
