"""
LLM-RL Hybrid Explainability for SupplyMind.

Uses LOCAL Ollama (qwen2.5:14b) — zero API limits, ~3-4 sec per explanation
on RTX 4080. After each RL action, decodes state to text and calls Ollama
for natural-language explanation of why the agent chose that action.

Pre-populates 50 common scenarios to cache/explanations.json for demo speed.

Usage:
    from rl.explainer import explain_action
    explanation = explain_action(obs, action, reward_components)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = CACHE_DIR / "explanations_cache.json"

ACTION_NAMES = {
    "do_nothing": "Wait and observe",
    "activate_backup_supplier": "Activate backup supplier",
    "reroute_shipment": "Reroute shipment via alternate port",
    "increase_safety_stock": "Increase safety stock buffer",
    "expedite_order": "Expedite order via air freight",
    "hedge_commodity": "Hedge commodity price exposure",
    "issue_supplier_alert": "Issue early warning alert to supplier",
}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _cache_key(state_summary: str, action: str) -> str:
    return hashlib.md5(f"{state_summary}|{action}".encode()).hexdigest()


def decode_state_to_text(obs) -> str:
    """Decode structured observation to natural language summary.

    Args:
        obs: SupplyMindObservation (Pydantic model from core env).

    Returns:
        Human-readable state summary (~200 words).
    """
    lines = []
    lines.append(f"Day {obs.current_day} of {obs.current_day + obs.days_remaining}.")

    # Financials
    fin = obs.financials
    lines.append(f"Budget: ${fin.budget_remaining:,.0f} of ${fin.budget_total:,.0f} remaining.")
    lines.append(f"Revenue lost so far: ${fin.cumulative_revenue_lost:,.0f}.")
    lines.append(f"Supply chain health: {fin.supply_chain_health_score:.0f}/100.")
    lines.append(f"Monte Carlo P50 loss: ${fin.monte_carlo_p50_loss:,.0f}, "
                 f"P95: ${fin.monte_carlo_p95_loss:,.0f}.")

    # Active disruptions
    if obs.active_signals:
        lines.append(f"\nActive disruptions ({len(obs.active_signals)}):")
        for sig in obs.active_signals[:5]:
            lines.append(f"  - {sig.disruption_type} ({sig.lifecycle_phase}): "
                         f"severity {sig.severity:.2f}, "
                         f"affecting {', '.join(sig.affected_node_ids[:3])}")

    # Critical nodes
    critical = [n for n in obs.node_statuses if not n.is_operational or n.current_risk_score > 0.5]
    if critical:
        lines.append(f"\nCritical nodes ({len(critical)}):")
        for n in critical[:5]:
            status = "OFFLINE" if not n.is_operational else f"risk={n.current_risk_score:.2f}"
            lines.append(f"  - {n.name} ({n.node_type}): {status}, "
                         f"inventory={n.inventory_days_cover:.0f}d")

    return "\n".join(lines)


def _build_prompt(state_text: str, action_type: str, target_node: str | None,
                  reward_components: dict | None) -> str:
    """Build Ollama prompt for action explanation."""
    action_desc = ACTION_NAMES.get(action_type, action_type)
    target_text = f" targeting {target_node}" if target_node else ""

    reward_text = ""
    if reward_components:
        reward_text = "\nReward breakdown: " + ", ".join(
            f"{k}={v:+.3f}" for k, v in reward_components.items()
        )

    return f"""You are a supply chain risk analyst AI explaining an RL agent's decision.

CURRENT STATE:
{state_text}

ACTION TAKEN: {action_desc}{target_text}
{reward_text}

Explain in 2-3 sentences WHY this action makes sense given the current supply chain state.
Focus on the risk-reward tradeoff and what would happen without this action.
Be specific about node names and financial impacts. Do not be generic."""


def explain_action(
    obs,
    action_type: str,
    target_node: str | None = None,
    reward_components: dict | None = None,
    use_ollama: bool = True,
    model_name: str = "qwen2.5:14b",
) -> str:
    """Generate natural-language explanation for an RL action.

    Checks cache first, then calls Ollama if available.

    Args:
        obs:               SupplyMindObservation (raw observation from env).
        action_type:       Action type string (e.g., "activate_backup_supplier").
        target_node:       Target node ID (optional).
        reward_components: Reward breakdown dict (optional).
        use_ollama:        Whether to call Ollama (False = cache/heuristic only).
        model_name:        Ollama model name.

    Returns:
        Explanation string.
    """
    state_text = decode_state_to_text(obs)
    cache = _load_cache()
    key = _cache_key(state_text[:200], action_type)

    # Check cache
    if key in cache:
        return cache[key]

    # Try Ollama
    if use_ollama:
        try:
            import ollama
            prompt = _build_prompt(state_text, action_type, target_node, reward_components)
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            explanation = response["message"]["content"].strip()

            # Cache the result
            cache[key] = explanation
            _save_cache(cache)
            return explanation

        except ImportError:
            logger.warning("ollama package not installed. Using heuristic explanation.")
        except Exception as e:
            logger.warning("Ollama call failed: %s. Using heuristic.", e)

    # Heuristic fallback (no API needed)
    return _heuristic_explanation(obs, action_type, target_node)


def _heuristic_explanation(obs, action_type: str, target_node: str | None) -> str:
    """Rule-based explanation when Ollama is unavailable."""
    fin = obs.financials
    health = fin.supply_chain_health_score
    budget_pct = (fin.budget_remaining / max(fin.budget_total, 1)) * 100

    explanations = {
        "do_nothing": (
            f"With health at {health:.0f}/100 and {budget_pct:.0f}% budget remaining, "
            f"the agent conserves resources. No immediate threats require action."
        ),
        "activate_backup_supplier": (
            f"Activating backup for {target_node} to mitigate disruption risk. "
            f"With P95 projected loss of ${fin.monte_carlo_p95_loss:,.0f}, "
            f"the $150K qualification cost is justified to protect revenue."
        ),
        "reroute_shipment": (
            f"Rerouting shipments away from {target_node} via alternate port. "
            f"The additional transit cost is offset by avoiding delays that "
            f"would increase cumulative losses beyond ${fin.cumulative_revenue_lost:,.0f}."
        ),
        "increase_safety_stock": (
            f"Building safety stock buffer at {target_node} to absorb "
            f"potential supply disruption. Current inventory cover may be insufficient "
            f"given active disruption severity."
        ),
        "expedite_order": (
            f"Expediting via air freight to {target_node} due to critical inventory levels. "
            f"The premium cost is warranted: stockout risk would trigger "
            f"SLA penalties exceeding ${fin.cumulative_penalty_fees:,.0f}."
        ),
        "hedge_commodity": (
            f"Hedging commodity exposure to lock in current prices. "
            f"Price volatility signals suggest further increases, "
            f"which would compound supply chain costs."
        ),
        "issue_supplier_alert": (
            f"Issuing early warning to {target_node} (free action). "
            f"Proactive alerting improves supplier response time and "
            f"earns the proactive bonus in grading."
        ),
    }
    return explanations.get(action_type, f"Action: {action_type} on {target_node}")


def pre_populate_cache(n_scenarios: int = 50) -> int:
    """Pre-populate explanation cache with common scenarios.

    Runs scripted agent on all 3 tasks and caches explanations
    for each action taken. Uses heuristic explanations (no Ollama needed).

    Returns number of explanations cached.
    """
    import sys
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from server.supply_environment import SupplyMindEnvironment
    from scripted_agent import choose_action

    env = SupplyMindEnvironment()
    cache = _load_cache()
    count = 0

    tasks = ["easy_typhoon_response", "medium_multi_front", "hard_cascading_crisis"]
    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        step = 0
        while not obs.done and count < n_scenarios:
            action = choose_action(obs, step)
            explanation = _heuristic_explanation(obs, action.action_type, action.target_node_id)
            state_text = decode_state_to_text(obs)
            key = _cache_key(state_text[:200], action.action_type)
            cache[key] = explanation
            count += 1
            obs = env.step(action)
            step += 1

    _save_cache(cache)
    logger.info("Pre-populated %d explanations to cache", count)
    return count
