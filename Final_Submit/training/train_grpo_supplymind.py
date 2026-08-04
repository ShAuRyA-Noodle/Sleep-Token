"""
train_grpo_supplymind.py — GRPO + TRL + Unsloth training of an LLM agent
on the SupplyMind OpenEnv environment.

Stack:
    - Unsloth: 4-bit NF4 quantization, 2x speed on T4
    - TRL: GRPOTrainer (group-relative policy optimization, no critic)
    - OpenEnv: HTTP client to SupplyMind environment

Run:
    python train_grpo_supplymind.py \\
        --env-url http://localhost:8000 \\
        --model unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit \\
        --steps 200 \\
        --output-dir ./grpo_supplymind_qwen05b

Or in Colab: open colab_train_grpo.ipynb and press Run All.

This script ALSO writes:
    - results/training_log.csv (per-step reward + loss)
    - results/reward_curve.png
    - results/loss_curve.png
    - results/baseline_vs_trained.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests

# Lazy imports — these are heavy and Colab-installed
def _lazy_imports():
    global FastLanguageModel, GRPOConfig, GRPOTrainer, Dataset, torch
    from unsloth import FastLanguageModel as _FLM  # type: ignore
    from trl import GRPOConfig as _GC, GRPOTrainer as _GT  # type: ignore
    from datasets import Dataset as _DS  # type: ignore
    import torch as _t
    FastLanguageModel = _FLM
    GRPOConfig = _GC
    GRPOTrainer = _GT
    Dataset = _DS
    torch = _t


# ---------------------------------------------------------------------------
# Env client (HTTP)
# ---------------------------------------------------------------------------

class SupplyMindClient:
    """Thin HTTP client for SupplyMind OpenEnv server."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task_id: str = "easy_typhoon_response", session_id: str | None = None,
              seed: int | None = None) -> dict:
        sid = session_id or f"grpo_{uuid.uuid4().hex[:8]}"
        body = {"task_id": task_id, "session_id": sid}
        if seed is not None:
            body["seed"] = seed
        r = self.session.post(f"{self.base}/reset", json=body, timeout=30)
        r.raise_for_status()
        return {"session_id": sid, **r.json()}

    def step(self, action: dict, session_id: str) -> dict:
        body = {"session_id": session_id, "action": action}
        r = self.session.post(f"{self.base}/step", json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self, session_id: str) -> dict:
        r = self.session.get(f"{self.base}/state", params={"session_id": session_id}, timeout=10)
        r.raise_for_status()
        return r.json()

    def grade(self, session_id: str) -> dict:
        r = self.session.post(f"{self.base}/grader", json={"session_id": session_id}, timeout=30)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Action parsing (bullet-proof — bad output → do_nothing fallback)
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = {
    "do_nothing", "issue_supplier_alert", "activate_backup_supplier",
    "reroute_shipment", "increase_safety_stock", "expedite_order",
    "hedge_commodity",
}


def parse_action(text: str) -> tuple[dict, bool]:
    """Extract action JSON from LLM completion. Returns (action, is_well_formed)."""
    if not text:
        return {"action_type": "do_nothing"}, False

    # Find first {...} JSON-ish block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if not m:
        return {"action_type": "do_nothing"}, False

    try:
        action = json.loads(m.group(0))
        if not isinstance(action, dict) or "action_type" not in action:
            return {"action_type": "do_nothing"}, False
        if action["action_type"] not in VALID_ACTION_TYPES:
            return {"action_type": "do_nothing"}, False
        return action, True
    except json.JSONDecodeError:
        return {"action_type": "do_nothing"}, False


# ---------------------------------------------------------------------------
# Prompt template (read by LLM agent)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a supply-chain risk manager. Given a state observation, output exactly ONE action as a JSON object on a single line.

Available action types:
- {"action_type": "do_nothing"}
- {"action_type": "issue_supplier_alert", "target_node_id": "<id>"}    — free
- {"action_type": "activate_backup_supplier", "target_node_id": "<id>", "backup_supplier_id": "<id>"}
- {"action_type": "increase_safety_stock", "target_node_id": "<id>", "additional_stock_days": <1-90>}
- {"action_type": "reroute_shipment", "target_node_id": "<id>", "reroute_via": ["<port_id>"]}
- {"action_type": "expedite_order", "target_node_id": "<id>", "expedite_mode": "air"}    (or "rail" / "express_sea")
- {"action_type": "hedge_commodity", "commodity": "oil", "hedge_amount_usd": <float>}

Decision principles:
1. Act EARLY during warning phase (highest reward).
2. PRIORITIZE high-revenue nodes.
3. Use FREE alerts for intel before spending.
4. Stay within budget.
"""


def build_prompt(observation: dict) -> str:
    """Format an env observation into a TRL-trainable prompt."""
    obs = observation.get("observation", observation)
    user_block = obs.get("compact_summary") or obs.get("situation_summary", "")
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_block}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def collect_prompts(client: SupplyMindClient, n_prompts: int, task_ids: list[str]) -> list[dict]:
    """Generate a dataset of (prompt, session_id, expected_grade_offset) tuples.

    For each task_id, reset env, step through with do_nothing, and capture the
    observation at each step as a training prompt.
    """
    rows = []
    for task in task_ids:
        n_per_task = max(1, n_prompts // len(task_ids))
        for _ in range(n_per_task):
            r = client.reset(task_id=task)
            sid = r["session_id"]
            obs = r.get("observation", r)
            for _step in range(min(8, n_per_task)):  # cap depth per session
                rows.append({
                    "prompt": build_prompt({"observation": obs}),
                    "task_id": task,
                })
                # Advance env with random/no-op so we get diverse states
                step_resp = client.step({"action_type": "do_nothing"}, session_id=sid)
                obs = step_resp.get("observation", step_resp)
                if step_resp.get("done"):
                    break
                if len(rows) >= n_prompts:
                    return rows
    return rows


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------

def make_reward_fn(client: SupplyMindClient, task_default: str = "easy_typhoon_response"):
    """Returns a TRL-compatible reward function.

    For each (prompt, completion) pair:
        1. Parse completion → action (do_nothing fallback if malformed)
        2. Reset a fresh session, step with the action, return env reward
        3. Apply format penalty if action wasn't well-formed
    """
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                action, is_well_formed = parse_action(completion)

                # Step a fresh session to get a clean reward signal
                r = client.reset(task_id=task_default)
                sid = r["session_id"]
                step_resp = client.step(action, session_id=sid)

                env_reward = float(step_resp.get("observation", {}).get("reward", 0.0))
                # Format bonus/penalty
                format_bonus = 0.05 if is_well_formed else -0.20
                # Free-action small bonus (encourages information gathering)
                free_bonus = 0.02 if action["action_type"] == "issue_supplier_alert" else 0.0

                rewards.append(env_reward + format_bonus + free_bonus)
            except Exception as e:
                # Defensive: any HTTP/parsing failure → strong negative
                print(f"[reward_fn] error: {e}")
                rewards.append(-1.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Plotting (committed PNGs)
# ---------------------------------------------------------------------------

def write_plots(log_csv: Path, output_dir: Path) -> None:
    """Generate the 3 committed PNGs from training_log.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, rewards, losses = [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row["mean_reward"]))
            losses.append(float(row.get("loss", "0") or 0.0))

    # Reward curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, rewards, "g-", linewidth=2, label="GRPO-trained Qwen-2.5-0.5B")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Mean reward (env.step()) per batch", fontsize=12)
    ax.set_title("SupplyMind GRPO Training — Reward Curve", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=120)
    plt.close()

    # Loss curve
    if any(losses):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, "r-", linewidth=2, label="GRPO loss")
        ax.set_xlabel("Training step", fontsize=12)
        ax.set_ylabel("GRPO loss", fontsize=12)
        ax.set_title("SupplyMind GRPO Training — Loss Curve", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=120)
        plt.close()

    print(f"[plots] wrote {output_dir}/reward_curve.png and {output_dir}/loss_curve.png")


def write_baseline_vs_trained(baseline_csv: Path, trained_csv: Path, output_dir: Path) -> None:
    """Side-by-side comparison plot — both on same axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def load(p):
        rs = []
        with open(p) as f:
            for row in csv.DictReader(f):
                rs.append(float(row["mean_reward"]))
        return rs

    base = load(baseline_csv)
    trnd = load(trained_csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    n = max(len(base), len(trnd))
    x = list(range(n))
    if base:
        ax.plot(x[:len(base)], base, "k--", linewidth=2, label=f"Baseline (n={len(base)} eps, mean={np.mean(base):.3f})")
    if trnd:
        ax.plot(x[:len(trnd)], trnd, "g-", linewidth=2, label=f"GRPO-trained (n={len(trnd)} eps, mean={np.mean(trnd):.3f})")
    ax.set_xlabel("Evaluation episode", fontsize=12)
    ax.set_ylabel("Episode return (sum of step rewards)", fontsize=12)
    ax.set_title("SupplyMind — Baseline vs GRPO-Trained Agent (same axes)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_vs_trained.png", dpi=120)
    plt.close()
    print(f"[plots] wrote {output_dir}/baseline_vs_trained.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default=os.environ.get("SUPPLYMIND_ENV_URL", "http://localhost:8000"))
    parser.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--output-dir", default="./grpo_supplymind_qwen05b")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Health-check env
    print(f"[env] checking {args.env_url}/health ...")
    r = requests.get(f"{args.env_url}/health", timeout=10)
    r.raise_for_status()
    print(f"[env] ok: {r.json()}")

    client = SupplyMindClient(args.env_url)

    # 2. Collect prompts
    print(f"[data] collecting {args.n_prompts} prompts across 3 tasks ...")
    rows = collect_prompts(client, args.n_prompts,
                           ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"])
    print(f"[data] collected {len(rows)} prompts")

    # 3. Lazy-import heavy ML deps
    _lazy_imports()
    train_dataset = Dataset.from_list(rows)

    # 4. Load model with Unsloth
    print(f"[model] loading {args.model} via Unsloth (4-bit NF4) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
    )

    # 5. GRPO config + reward fn
    reward_fn = make_reward_fn(client, task_default="easy_typhoon_response")
    config = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=args.num_generations,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        max_steps=args.steps,
        save_steps=max(50, args.steps // 4),
        logging_steps=5,
        bf16=True,
        report_to="none",
        max_prompt_length=1024,
        max_completion_length=128,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
    )

    # 6. Train (with logging tee)
    log_csv_path = results_dir / "training_log.csv"
    print(f"[train] starting GRPO for {args.steps} steps ...")
    t0 = time.time()
    train_result = trainer.train()
    t1 = time.time()
    print(f"[train] done in {t1 - t0:.1f}s")

    # 7. Dump training log
    state_log = trainer.state.log_history
    with open(log_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "loss"])
        for entry in state_log:
            step = entry.get("step", entry.get("global_step", 0))
            r = entry.get("reward", entry.get("rewards/mean", 0.0))
            l = entry.get("loss", 0.0)
            w.writerow([step, r, l])
    print(f"[train] wrote {log_csv_path}")

    # 8. Save adapter
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"[save] adapter at {final_dir}")

    # 9. Plots
    write_plots(log_csv_path, results_dir)

    # 10. Done
    print(f"\n✅ Training complete in {t1-t0:.1f}s")
    print(f"   Adapter: {final_dir}")
    print(f"   Log:     {log_csv_path}")
    print(f"   Plots:   {results_dir}/")


if __name__ == "__main__":
    main()
