"""
run_baseline.py — Collect untrained-baseline performance for the
baseline_vs_trained.png comparison.

Runs an UNTRAINED Qwen-2.5-0.5B (or any HF model) against the SupplyMind
environment for N episodes and saves per-episode returns.

Run:
    python run_baseline.py --env-url http://localhost:8000 --episodes 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path

import requests

# Reuse the parser + prompt template from the trainer
from train_grpo_supplymind import (
    SupplyMindClient,
    build_prompt,
    parse_action,
    SYSTEM_PROMPT,
)


def _lazy_imports():
    global AutoTokenizer, AutoModelForCausalLM, torch
    from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM  # type: ignore
    import torch as _t
    AutoTokenizer = _AT
    AutoModelForCausalLM = _AM
    torch = _t


def run_episode(client: SupplyMindClient, model, tokenizer, task_id: str, max_steps: int = 30) -> float:
    """Run one episode with the LLM as agent, return total reward."""
    r = client.reset(task_id=task_id)
    sid = r["session_id"]
    obs = r.get("observation", r)
    total_reward = 0.0
    for step in range(max_steps):
        prompt = build_prompt({"observation": obs})
        # Tokenize + generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action, _ = parse_action(completion)
        step_resp = client.step(action, session_id=sid)
        obs = step_resp.get("observation", step_resp)
        total_reward += float(obs.get("reward", 0.0))
        if step_resp.get("done"):
            break
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default=os.environ.get("SUPPLYMIND_ENV_URL", "http://localhost:8000"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task", default="easy_typhoon_response")
    parser.add_argument("--output", default="./results/baseline_log.csv")
    parser.add_argument("--adapter", default=None,
                        help="Optional path to LoRA adapter (for trained-agent eval)")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    _lazy_imports()

    client = SupplyMindClient(args.env_url)

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  device_map="auto")
    if args.adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[load] applied adapter {args.adapter}")

    print(f"[run] {args.episodes} episodes on {args.task} ...")
    returns = []
    t0 = time.time()
    for i in range(args.episodes):
        ret = run_episode(client, model, tokenizer, args.task)
        returns.append(ret)
        print(f"  ep {i+1:3d}: return = {ret:+.3f}")
    t1 = time.time()

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "mean_reward", "label"])
        for i, r in enumerate(returns):
            w.writerow([i, r, args.label])

    import statistics
    print(f"\n✅ {args.label}: {args.episodes} episodes in {t1-t0:.1f}s")
    print(f"   mean return = {statistics.mean(returns):+.3f}")
    print(f"   stdev       = {statistics.stdev(returns):+.3f}" if len(returns) > 1 else "")
    print(f"   wrote {args.output}")


if __name__ == "__main__":
    main()
