"""
reward_function.py — The verifier-first reward function for SupplyMind.

Design principles (from the hackathon brief):
1. Verifier first — programmatic, hard to fool, returns clear signal
2. Layered defense — multiple independent components, AI must pass all
3. Process + outcome — env.step() per-step + env.grade() episode-level
4. Anti-hacking — format check, length check, action-validity check

This file is imported by train_grpo_supplymind.py and is also runnable
standalone for debugging:

    python reward_function.py --env-url http://localhost:8000 --action '{"action_type": "do_nothing"}'
"""
from __future__ import annotations

import json
import re
from typing import Optional

VALID_ACTION_TYPES = {
    "do_nothing", "issue_supplier_alert", "activate_backup_supplier",
    "reroute_shipment", "increase_safety_stock", "expedite_order",
    "hedge_commodity",
}

REQUIRED_FIELDS_BY_ACTION = {
    "do_nothing": [],
    "issue_supplier_alert": ["target_node_id"],
    "activate_backup_supplier": ["target_node_id", "backup_supplier_id"],
    "increase_safety_stock": ["target_node_id", "additional_stock_days"],
    "reroute_shipment": ["target_node_id", "reroute_via"],
    "expedite_order": ["target_node_id", "expedite_mode"],
    "hedge_commodity": ["commodity", "hedge_amount_usd"],
}


def parse_action(text: str) -> tuple[dict, dict]:
    """
    Returns (action, diagnostics).

    diagnostics: {
        "is_json": bool,
        "has_action_type": bool,
        "is_valid_type": bool,
        "has_required_fields": bool,
        "format_score": float in [0, 1],
    }
    """
    diag = {"is_json": False, "has_action_type": False, "is_valid_type": False,
            "has_required_fields": False, "format_score": 0.0}

    if not text:
        return {"action_type": "do_nothing"}, diag

    # Find JSON block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if not m:
        return {"action_type": "do_nothing"}, diag

    try:
        action = json.loads(m.group(0))
        diag["is_json"] = True
    except json.JSONDecodeError:
        return {"action_type": "do_nothing"}, diag

    if not isinstance(action, dict) or "action_type" not in action:
        return {"action_type": "do_nothing"}, diag
    diag["has_action_type"] = True

    if action["action_type"] not in VALID_ACTION_TYPES:
        return {"action_type": "do_nothing"}, diag
    diag["is_valid_type"] = True

    required = REQUIRED_FIELDS_BY_ACTION[action["action_type"]]
    diag["has_required_fields"] = all(f in action for f in required)

    diag["format_score"] = sum([
        diag["is_json"], diag["has_action_type"],
        diag["is_valid_type"], diag["has_required_fields"]
    ]) / 4.0

    return action, diag


def composite_reward(env_reward: float, diag: dict, completion_text: str = "") -> float:
    """
    Compose the GRPO training reward from:
      - env.step() reward (the main signal)
      - format quality (programmatic check)
      - length penalty (anti-spam)

    All three must be passed for full reward.
    """
    base = float(env_reward)

    # Format component (-0.20 if malformed, +0.05 if perfect)
    fmt_bonus = -0.20 + 0.25 * diag["format_score"]

    # Length penalty — anti-spam (>400 tokens approximately)
    n_chars = len(completion_text or "")
    if n_chars > 1500:
        length_penalty = -0.15
    elif n_chars > 800:
        length_penalty = -0.05
    else:
        length_penalty = 0.0

    return base + fmt_bonus + length_penalty


def cli():
    import argparse
    import requests
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--action", required=True, help="JSON string of action")
    parser.add_argument("--task", default="easy_typhoon_response")
    args = parser.parse_args()

    action, diag = parse_action(args.action)
    print(f"parsed action: {action}")
    print(f"diagnostics: {json.dumps(diag, indent=2)}")

    # Round-trip through env
    sid = "reward_debug"
    requests.post(f"{args.env_url}/reset", json={"task_id": args.task, "session_id": sid})
    r = requests.post(f"{args.env_url}/step", json={"session_id": sid, "action": action}).json()
    env_reward = r.get("observation", {}).get("reward", 0.0)
    print(f"env reward: {env_reward}")

    composite = composite_reward(env_reward, diag, args.action)
    print(f"composite reward: {composite}")


if __name__ == "__main__":
    cli()
