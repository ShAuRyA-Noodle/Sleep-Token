"""
EBITDA impact model for supply chain disruptions.

Richer financial impact calculation than the core engine:
  - lost_margin: revenue_per_day * gross_margin
  - expedite_premium: emergency logistics cost multiplier
  - sla_penalties: contractual penalty fees
  - reputation_cost: brand damage estimate (5% of daily revenue)

Returns per-day and total impact with full breakdown.

Sources: McKinsey/CSCMP industry cost benchmarks.
"""

from __future__ import annotations

from typing import Any


# Industry-average financial parameters (McKinsey Supply Chain benchmarks)
DEFAULT_FINANCIALS = {
    "gross_margin": 0.35,           # 35% average across manufacturing
    "expedite_cost_multiplier": 3.0,  # 3x normal logistics for emergency
    "expedite_fraction": 0.30,       # 30% of daily revenue needs expediting
    "sla_penalty_per_day": 25_000,   # $25K/day average SLA penalty
    "sla_buffer_days": 3,            # 3-day grace period
    "reputation_cost_fraction": 0.05,  # 5% of daily revenue (conservative)
}


def calculate_ebitda_impact(
    disruption: dict[str, Any],
    company_financials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate daily and total EBITDA impact per disrupted node.

    Args:
        disruption: Dict with:
            - revenue_at_risk: float (annual revenue through this node)
            - expected_duration_days: int
            - delay_days: int (current delay to customers)
            - affected_customers: list of dicts with sla_penalty_per_day, sla_buffer_days
            - severity: float (0-1)
        company_financials: Optional overrides for DEFAULT_FINANCIALS.

    Returns:
        Dict with daily_ebitda_impact, total_estimate, and breakdown.
    """
    fin = {**DEFAULT_FINANCIALS, **(company_financials or {})}

    revenue_at_risk = disruption.get("revenue_at_risk", 0)
    duration = disruption.get("expected_duration_days", 30)
    delay_days = disruption.get("delay_days", 0)
    severity = disruption.get("severity", 0.5)
    affected_customers = disruption.get("affected_customers", [])

    # Daily revenue through disrupted node
    revenue_per_day = revenue_at_risk / 365

    # 1. Lost margin: direct revenue loss * gross margin * severity
    lost_margin = revenue_per_day * fin["gross_margin"] * severity

    # 2. Expedite premium: emergency logistics costs
    expedite_premium = (
        fin["expedite_cost_multiplier"]
        * revenue_per_day
        * fin["expedite_fraction"]
        * severity
    )

    # 3. SLA penalties: sum across affected customers
    sla_penalties = 0.0
    if affected_customers:
        for customer in affected_customers:
            buffer = customer.get("sla_buffer_days", fin["sla_buffer_days"])
            if delay_days > buffer:
                penalty = customer.get("sla_penalty_per_day", fin["sla_penalty_per_day"])
                sla_penalties += penalty
    else:
        # Estimate: penalty kicks in after buffer period
        if delay_days > fin["sla_buffer_days"]:
            sla_penalties = fin["sla_penalty_per_day"]

    # 4. Reputation cost: brand/relationship damage
    reputation_cost = revenue_per_day * fin["reputation_cost_fraction"] * severity

    daily_impact = lost_margin + expedite_premium + sla_penalties + reputation_cost
    total_estimate = daily_impact * duration

    return {
        "daily_ebitda_impact": round(daily_impact, 2),
        "total_estimate": round(total_estimate, 2),
        "duration_days": duration,
        "severity": severity,
        "breakdown": {
            "lost_margin": round(lost_margin, 2),
            "expedite_premium": round(expedite_premium, 2),
            "sla_penalties": round(sla_penalties, 2),
            "reputation_cost": round(reputation_cost, 2),
        },
        "breakdown_pct": {
            "lost_margin": round(lost_margin / max(daily_impact, 1) * 100, 1),
            "expedite_premium": round(expedite_premium / max(daily_impact, 1) * 100, 1),
            "sla_penalties": round(sla_penalties / max(daily_impact, 1) * 100, 1),
            "reputation_cost": round(reputation_cost / max(daily_impact, 1) * 100, 1),
        },
    }


def estimate_cascade_impact(
    nodes: list[dict[str, Any]],
    severity: float = 0.5,
    duration_days: int = 30,
) -> dict[str, Any]:
    """Estimate total EBITDA impact across multiple disrupted nodes.

    Args:
        nodes:         List of disrupted node dicts (each needs revenue_at_risk).
        severity:      Common severity level.
        duration_days: Expected disruption duration.

    Returns:
        Aggregated impact summary.
    """
    total_daily = 0.0
    total_total = 0.0
    node_impacts = []

    for node in nodes:
        disruption = {
            "revenue_at_risk": node.get("annual_spend", node.get("revenue_at_risk", 0)),
            "expected_duration_days": duration_days,
            "delay_days": node.get("delay_days", 5),
            "severity": severity,
            "affected_customers": [],
        }
        impact = calculate_ebitda_impact(disruption)
        total_daily += impact["daily_ebitda_impact"]
        total_total += impact["total_estimate"]
        node_impacts.append({
            "node_id": node.get("id", node.get("node_id", "unknown")),
            "name": node.get("name", "unknown"),
            **impact,
        })

    return {
        "total_daily_impact": round(total_daily, 2),
        "total_impact": round(total_total, 2),
        "nodes_affected": len(nodes),
        "duration_days": duration_days,
        "severity": severity,
        "per_node": node_impacts,
    }
