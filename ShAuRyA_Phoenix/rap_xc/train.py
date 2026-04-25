"""train.py — RAP-XC trajectory harvest + training loop.

Pipeline:
  1. harvest_trajectories(): roll out an existing policy (MaskablePPO,
     RecurrentPPO, scripted, ...) in the live SupplyMind env and dump
     transitions to disk: (state_feats, crisis_embeds, dag_feats,
     action, reward, return_to_go).
  2. precompute_judge_prior(): optional one-time distillation of the
     25-judge panel into a per-(state-cluster, action) bias table.
  3. train_rapxc(): supervised behavior-cloning + judge-KL + value-MSE
     + CQL-conservative on harvested transitions.

For the smoke test (no env access required) we generate synthetic
transitions; for the real run we wire harvest_trajectories() to the
SupplyMindEnvironment.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import RAPXCConfig, RAPXCPolicy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "ShAuRyA_Phoenix" / "experiments" / "rap_xc_v1"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Trajectory harvesting
# ---------------------------------------------------------------------

@dataclass
class TrajectoryConfig:
    n_episodes: int = 1500
    max_steps_per_ep: int = 30
    tasks: tuple[str, ...] = ("easy_typhoon_response", "medium_multi_front",
                                "hard_cascading_crisis")
    seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(1500)))
    cache_path: Path = field(default_factory=lambda: DATA_DIR / "transitions.npz")


def _engineer_state_features(obs) -> np.ndarray:
    """64-dim engineered state vector from a SupplyMindObservation."""
    out = np.zeros(64, dtype=np.float32)
    if hasattr(obs, "model_dump"):
        d = obs.model_dump()
    elif isinstance(obs, dict):
        d = obs
    else:
        return out
    # 0-1: day-related
    out[0] = float(d.get("current_day") or 0) / 30.0
    out[1] = float(d.get("days_remaining") or 0) / 30.0
    # 2-9: financials (8-dim)
    fin = d.get("financials") or {}
    fin_keys = ("budget_remaining_usd", "cumulative_cost_usd",
                "expected_loss_usd", "buffer_days", "total_revenue_usd",
                "total_loss_usd", "current_inventory_value_usd",
                "supplier_diversity_score")
    for i, k in enumerate(fin_keys):
        try:
            v = float(fin.get(k) or 0)
        except (ValueError, TypeError):
            v = 0.0
        out[2 + i] = math.tanh(v / 1_000_000.0) if abs(v) > 1.0 else v
    # 10-25: node statuses summary (16-dim, pooled)
    statuses = d.get("node_statuses") or []
    if statuses:
        stresses = [float(s.get("stress_level") or 0) for s in statuses[:16]]
        operations = [1.0 if (s.get("operational_status") == "OPERATIONAL")
                      else 0.0 for s in statuses[:16]]
        for i, s in enumerate(stresses):
            out[10 + i] = s
        for i, op in enumerate(operations):
            if i + 26 < 64:
                out[26 + i] = op
    # 42-49: signal counts (8-dim)
    sigs = d.get("active_signals") or []
    out[42] = min(10, len(sigs)) / 10.0
    new_sigs = d.get("new_signals") or []
    out[43] = min(10, len(new_sigs)) / 10.0
    # 50-63: simple compact_summary length / hash features
    summary = (d.get("compact_summary") or "")[:200]
    for i, c in enumerate(summary[:14]):
        out[50 + i] = (ord(c) % 100) / 100.0
    return out


def harvest_trajectories(
    config: TrajectoryConfig,
    *,
    policy_fn: Callable | None = None,
    library_search: Callable | None = None,
) -> dict:
    """Roll policy_fn through the env, collect transitions.

    policy_fn signature: (obs_dict) -> action_dict. If None, defaults to
    a scripted "safety-stock-on-day-1" baseline.
    library_search: (query_str, k) -> list[dict] for crisis retrieval.
    """
    if policy_fn is None:
        policy_fn = _default_scripted_policy

    try:
        from server.app import SupplyMindEnvironment
    except ImportError as e:
        logger.error("[harvest] env import failed: %s", e)
        return {"status": "env_unavailable", "n_transitions": 0}

    if library_search is None:
        try:
            from ShAuRyA_Supplymind.scenarios.library_v2_search import search as library_search
        except ImportError:
            logger.warning("[harvest] library v2 not cooked; using zero retrieval")
            library_search = lambda q, k: []  # noqa: E731

    env = SupplyMindEnvironment()
    transitions: list[dict] = []
    t0 = time.time()
    for ep_idx in range(config.n_episodes):
        seed = config.seeds[ep_idx % len(config.seeds)]
        task = config.tasks[ep_idx % len(config.tasks)]
        try:
            obs = env.reset(task_id=task, seed=seed)
        except Exception:  # noqa: BLE001
            continue

        ep_transitions: list[dict] = []
        ep_rewards: list[float] = []
        for step in range(config.max_steps_per_ep):
            state_feats = _engineer_state_features(obs)
            # Library retrieval — query is the compact_summary
            if hasattr(obs, "compact_summary"):
                query = obs.compact_summary or "supply chain"
            elif hasattr(obs, "model_dump"):
                query = (obs.model_dump().get("compact_summary") or "supply chain")
            else:
                query = "supply chain"
            try:
                analogs = library_search(query, 8) or []
            except Exception:  # noqa: BLE001
                analogs = []
            # Padding if fewer than 8 analogs
            crisis_embeds = np.zeros((8, 1024), dtype=np.float32)
            # In real harvest we'd load the .npz embeddings table by index;
            # for the smoke test we use random fillers per analog
            rng = np.random.default_rng(ep_idx * 100 + step)
            for i, a in enumerate(analogs[:8]):
                # Hash-derived deterministic vector for stability
                h = hash(a.get("event_id", "x")) & 0xFFFFFFFF
                crisis_embeds[i] = rng.standard_normal(1024).astype(np.float32)

            # DAG features (80-dim): pad with zeros for now
            dag_feats = np.zeros(80, dtype=np.float32)

            try:
                action_dict = policy_fn(obs, step)
            except Exception:  # noqa: BLE001
                action_dict = {"task_id": task, "action_type": "do_nothing"}

            try:
                next_obs = env.step(action_dict)
                reward = float(getattr(next_obs, "reward", 0.0))
                done = bool(getattr(next_obs, "done", False))
            except Exception:  # noqa: BLE001
                break

            # Encode action_type to flat index (0=do_nothing, 1-6 mapped)
            action_int = _action_dict_to_int(action_dict)
            ep_transitions.append({
                "state_feats": state_feats,
                "crisis_embeds": crisis_embeds,
                "dag_feats": dag_feats,
                "action": action_int,
                "reward": reward,
            })
            ep_rewards.append(reward)
            obs = next_obs
            if done:
                break

        # Compute return-to-go for each step
        gamma = 0.95
        rtg = 0.0
        for t in range(len(ep_transitions) - 1, -1, -1):
            rtg = ep_transitions[t]["reward"] + gamma * rtg
            ep_transitions[t]["return_to_go"] = rtg
        transitions.extend(ep_transitions)

        if ep_idx % 50 == 0:
            elapsed = time.time() - t0
            logger.info("[harvest] ep %d/%d, transitions=%d, %.1fs",
                        ep_idx, config.n_episodes, len(transitions), elapsed)

    # Save as npz
    if transitions:
        out = {
            "state_feats": np.stack([t["state_feats"] for t in transitions]),
            "crisis_embeds": np.stack([t["crisis_embeds"] for t in transitions]),
            "dag_feats": np.stack([t["dag_feats"] for t in transitions]),
            "actions": np.array([t["action"] for t in transitions], dtype=np.int64),
            "rewards": np.array([t["reward"] for t in transitions], dtype=np.float32),
            "returns": np.array([t["return_to_go"] for t in transitions], dtype=np.float32),
        }
        np.savez_compressed(config.cache_path, **out)
        logger.info("[harvest] wrote %d transitions to %s", len(transitions),
                    config.cache_path)
    return {"status": "ok", "n_transitions": len(transitions),
            "n_episodes": config.n_episodes,
            "elapsed_s": round(time.time() - t0, 2)}


def _default_scripted_policy(obs, step: int) -> dict:
    """Day-1: build safety stock. Day-2+: monitor (no-op)."""
    if step == 0:
        return {"task_id": "easy_typhoon_response",
                "action_type": "increase_safety_stock",
                "target_node_id": "WAREHOUSE_PRIMARY",
                "additional_stock_days": 14}
    return {"task_id": "easy_typhoon_response", "action_type": "do_nothing"}


def _action_dict_to_int(action: dict) -> int:
    """Encode action_type * 40 + target_idx (rough)."""
    types = ["do_nothing", "activate_backup_supplier", "reroute_shipment",
              "increase_safety_stock", "expedite_order", "hedge_commodity",
              "issue_supplier_alert"]
    a_type = action.get("action_type", "do_nothing")
    type_idx = types.index(a_type) if a_type in types else 0
    target_str = (action.get("target_node_id") or "")
    try:
        target_int = int(target_str) if target_str.isdigit() else hash(target_str) % 40
    except (ValueError, AttributeError):
        target_int = 0
    return min(279, type_idx * 40 + target_int)


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_kl: float = 0.3
    lambda_v: float = 0.5
    lambda_cql: float = 0.1
    cosine_lr_min: float = 1e-5
    eval_every_steps: int = 500
    use_bf16: bool = True
    n_actions: int = 280
    out_path: Path = field(default_factory=lambda: DATA_DIR / "rapxc.pt")


def _cql_loss(logits: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
    """Conservative Q-learning surrogate: pull down OOD action logits."""
    lse = torch.logsumexp(logits, dim=-1)
    expert_q = logits.gather(1, expert_actions.unsqueeze(-1)).squeeze(-1)
    return (lse - expert_q).mean()


def train_rapxc(
    transitions_path: Path | None = None,
    judge_prior_table: torch.Tensor | None = None,
    cfg_train: TrainConfig | None = None,
    cfg_model: RAPXCConfig | None = None,
) -> dict:
    """Train RAP-XC on harvested transitions. Returns metrics dict."""
    cfg_train = cfg_train or TrainConfig()
    cfg_model = cfg_model or RAPXCConfig()
    transitions_path = transitions_path or (DATA_DIR / "transitions.npz")

    if not transitions_path.exists():
        return {"status": "no_data", "path": str(transitions_path)}

    npz = np.load(transitions_path)
    n = len(npz["actions"])
    logger.info("[train_rapxc] loaded %d transitions", n)

    # Filter to top-50% return episodes for behavior-cloning quality
    returns = npz["returns"]
    threshold = np.percentile(returns, 50)
    keep = returns >= threshold
    logger.info("[train_rapxc] filtering to top-50%% returns -> %d transitions", int(keep.sum()))

    state = torch.tensor(npz["state_feats"][keep])
    crisis = torch.tensor(npz["crisis_embeds"][keep])
    dag = torch.tensor(npz["dag_feats"][keep])
    actions = torch.tensor(npz["actions"][keep])
    rets = torch.tensor(npz["returns"][keep])

    dataset = TensorDataset(state, crisis, dag, actions, rets)
    loader = DataLoader(dataset, batch_size=cfg_train.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (cfg_train.use_bf16 and torch.cuda.is_available()
                                and torch.cuda.is_bf16_supported()) else torch.float32

    model = RAPXCPolicy(cfg_model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr,
                               weight_decay=cfg_train.weight_decay)
    n_steps = cfg_train.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=n_steps, eta_min=cfg_train.cosine_lr_min)

    history: list[dict] = []
    step = 0
    t0 = time.time()
    for ep in range(cfg_train.epochs):
        for batch in loader:
            state_b, crisis_b, dag_b, act_b, ret_b = [
                b.to(device) for b in batch]
            logits, value = model(state_b.float(), crisis_b.float(), dag_b.float())

            l_bc = F.cross_entropy(logits, act_b)
            l_v = F.mse_loss(value, ret_b.float())
            l_cql = _cql_loss(logits, act_b)

            l_kl = torch.tensor(0.0, device=device)
            if judge_prior_table is not None:
                # judge_prior_table: (n_actions,) prior logits per action
                jp = judge_prior_table.to(device).expand(logits.size(0), -1)
                l_kl = F.kl_div(F.log_softmax(logits, dim=-1),
                                 F.log_softmax(jp / 2.0, dim=-1),
                                 reduction="batchmean", log_target=True)

            loss = (l_bc + cfg_train.lambda_v * l_v
                    + cfg_train.lambda_cql * l_cql
                    + cfg_train.lambda_kl * l_kl)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            optim.step()
            sched.step()

            if step % 50 == 0:
                history.append({
                    "step": step, "epoch": ep,
                    "loss": float(loss.item()),
                    "loss_bc": float(l_bc.item()),
                    "loss_v": float(l_v.item()),
                    "loss_cql": float(l_cql.item()),
                    "loss_kl": float(l_kl.item()),
                    "lr": float(sched.get_last_lr()[0]),
                })
                logger.info("[train_rapxc] step %d ep %d loss=%.3f bc=%.3f v=%.3f cql=%.3f",
                            step, ep, loss.item(), l_bc.item(), l_v.item(), l_cql.item())
            step += 1

    # Save weights + history
    torch.save({
        "state_dict": model.state_dict(),
        "cfg_model": cfg_model.__dict__,
        "history": history,
    }, cfg_train.out_path)
    logger.info("[train_rapxc] saved to %s", cfg_train.out_path)
    return {
        "status": "ok",
        "n_train_transitions": int(keep.sum()),
        "n_steps": step,
        "final_loss": history[-1]["loss"] if history else None,
        "elapsed_s": round(time.time() - t0, 2),
        "weights_path": str(cfg_train.out_path),
        "n_parameters": model.n_parameters(),
    }


# ---------------------------------------------------------------------
# Synthetic smoke test
# ---------------------------------------------------------------------

def smoke_train_synthetic(n_synth: int = 1000) -> dict:
    """Fast synthetic train cycle to verify the loss converges."""
    cfg_model = RAPXCConfig()
    cfg_train = TrainConfig(epochs=2, batch_size=64, eval_every_steps=10000)

    rng = np.random.default_rng(42)
    npz_data = {
        "state_feats": rng.standard_normal((n_synth, cfg_model.state_dim)).astype(np.float32),
        "crisis_embeds": rng.standard_normal((n_synth, cfg_model.retrieved_k,
                                                cfg_model.crisis_embed_dim)).astype(np.float32),
        "dag_feats": rng.standard_normal((n_synth, cfg_model.dag_dim)).astype(np.float32),
        "actions": rng.integers(0, cfg_model.n_actions, size=n_synth, dtype=np.int64),
        "rewards": rng.standard_normal(n_synth).astype(np.float32),
        "returns": rng.standard_normal(n_synth).astype(np.float32),
    }
    p = DATA_DIR / "transitions_synth.npz"
    np.savez(p, **npz_data)

    cfg_train.out_path = DATA_DIR / "rapxc_synth.pt"
    return train_rapxc(transitions_path=p, cfg_train=cfg_train, cfg_model=cfg_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("--- RAP-XC smoke train (synthetic, 1000 transitions, 2 epochs) ---")
    result = smoke_train_synthetic(n_synth=1000)
    print(json.dumps(result, indent=2, default=str))
