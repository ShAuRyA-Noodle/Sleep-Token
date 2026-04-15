"""
Safety stock recommendation engine.

Risk-adjusted inventory buffer calculation:
  - risk_adjusted_lead_time = base * (1 + disruption_prob * avg_duration / base)
  - buffer = risk_adjusted_lead_time * daily_demand * risk_multiplier

Risk multipliers:
  - conservative: 2.5 (protect against 99% of scenarios)
  - moderate: 1.5 (balance cost and protection)
  - aggressive: 1.0 (minimum viable buffer)

Returns: recommended_units, cost, days_of_cover.

Source: CSCMP Supply Chain Risk Management framework.
"""

from __future__ import annotations

from typing import Any


RISK_MULTIPLIERS = {
    "conservative": 2.5,
    "moderate": 1.5,
    "aggressive": 1.0,
}


def recommend_safety_stock(
    component: dict[str, Any],
    risk_tolerance: str = "moderate",
) -> dict[str, Any]:
    """Compute risk-adjusted safety stock recommendation.

    Args:
        component: Dict with:
            - base_lead_time_days: Normal supplier lead time.
            - annual_demand_units: Annual unit demand.
            - unit_cost: Cost per unit ($).
            - disruption_probability: Probability of disruption (0-1).
            - avg_disruption_duration_days: Expected disruption length.
            - current_inventory_days: Current inventory cover in days.
            - name: Component/node name.
        risk_tolerance: "conservative", "moderate", or "aggressive".

    Returns:
        Dict with recommended buffer, cost, and analysis.
    """
    base_lt = component.get("base_lead_time_days", 30)
    annual_demand = component.get("annual_demand_units", 10_000)
    unit_cost = component.get("unit_cost", 100)
    disruption_prob = component.get("disruption_probability", 0.1)
    avg_duration = component.get("avg_disruption_duration_days", 14)
    current_inv = component.get("current_inventory_days", 10)

    # Risk-adjusted lead time
    risk_adjusted_lt = base_lt * (
        1 + disruption_prob * avg_duration / max(base_lt, 1)
    )

    # Daily demand
    daily_demand = annual_demand / 365

    # Risk multiplier
    multiplier = RISK_MULTIPLIERS.get(risk_tolerance, 1.5)

    # Recommended buffer units
    buffer_units = risk_adjusted_lt * daily_demand * multiplier

    # Cost
    holding_cost_rate = 0.25  # 25% annual carrying cost (industry standard)
    buffer_cost = buffer_units * unit_cost * holding_cost_rate

    # Days of cover
    days_of_cover = buffer_units / max(daily_demand, 0.01)

    # Gap analysis vs current
    current_units = current_inv * daily_demand
    gap_units = max(0, buffer_units - current_units)
    gap_cost = gap_units * unit_cost

    return {
        "component_name": component.get("name", "unknown"),
        "risk_tolerance": risk_tolerance,
        "recommended_buffer_units": round(buffer_units),
        "annual_holding_cost": round(buffer_cost, 2),
        "days_of_cover": round(days_of_cover, 1),
        "current_inventory_days": current_inv,
        "gap_units": round(gap_units),
        "gap_procurement_cost": round(gap_cost, 2),
        "analysis": {
            "base_lead_time_days": base_lt,
            "risk_adjusted_lead_time_days": round(risk_adjusted_lt, 1),
            "disruption_probability": disruption_prob,
            "avg_disruption_duration_days": avg_duration,
            "daily_demand": round(daily_demand, 1),
            "risk_multiplier": multiplier,
        },
    }


def recommend_all(
    components: list[dict[str, Any]],
    risk_tolerance: str = "moderate",
) -> dict[str, Any]:
    """Generate safety stock recommendations for all components.

    Returns aggregate summary and per-component recommendations.
    """
    recommendations = [
        recommend_safety_stock(comp, risk_tolerance) for comp in components
    ]

    total_cost = sum(r["annual_holding_cost"] for r in recommendations)
    total_gap_cost = sum(r["gap_procurement_cost"] for r in recommendations)

    return {
        "risk_tolerance": risk_tolerance,
        "total_annual_holding_cost": round(total_cost, 2),
        "total_gap_procurement_cost": round(total_gap_cost, 2),
        "components": len(recommendations),
        "recommendations": recommendations,
    }
