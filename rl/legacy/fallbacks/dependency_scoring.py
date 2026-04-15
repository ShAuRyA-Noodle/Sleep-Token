"""
Dependency scoring per supply chain node.

Quantifies how critical each supplier is via 4 components:
  - single_source_penalty (max 40): no backup = catastrophic risk
  - revenue_exposure (max 30): downstream revenue concentration
  - lead_time_risk (max 15): long lead times = slow recovery
  - geo_concentration (max 15): country-level concentration risk

Score range: 0-100. Higher = more critical.

Source: McKinsey Global Institute Supply Chain Risk Framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Country concentration risk scores (based on World Bank governance indicators)
# Higher = more concentrated risk in that country's supply chain role
COUNTRY_CONCENTRATION = {
    "Taiwan": 14,    # TSMC dominance, geopolitical risk
    "China": 12,     # Manufacturing concentration, trade policy risk
    "South Korea": 10,  # Samsung/SK Hynix, geopolitical proximity
    "Japan": 7,      # Earthquake risk, but diversified economy
    "US": 4,         # Stable governance, diversified
    "Germany": 4,    # Stable, diversified
    "Vietnam": 9,    # Growing but infrastructure risk
    "India": 8,      # Governance variability, infrastructure
    "Malaysia": 8,   # Semiconductor packaging concentration
    "Singapore": 3,  # Stable hub
    "Netherlands": 3,  # ASML concentration
}


def dependency_score(
    node: dict[str, Any],
    total_revenue: float,
) -> dict[str, Any]:
    """Compute dependency criticality score for a supply chain node.

    Args:
        node:          Node dict with keys: single_source, annual_spend,
                       lead_time_days, country, node_type, name.
        total_revenue: Total downstream revenue at risk.

    Returns:
        Dict with total_score and component breakdown.
    """
    # 1. Single source penalty (max 40)
    single_source = node.get("single_source", False)
    has_backup = bool(node.get("backup_supplier_ids"))
    if single_source and not has_backup:
        single_source_penalty = 40
    elif single_source and has_backup:
        single_source_penalty = 15  # Mitigated but still concentrated
    else:
        single_source_penalty = 0

    # 2. Revenue exposure (max 30)
    annual_spend = node.get("annual_spend", 0)
    if total_revenue > 0:
        revenue_pct = (annual_spend / total_revenue) * 100
    else:
        revenue_pct = 0
    revenue_exposure = min(30, revenue_pct)

    # 3. Lead time risk (max 15): longer lead time = slower recovery
    lead_time_days = node.get("lead_time_days", 0)
    lead_time_risk = min(15, lead_time_days / 7 * 5)

    # 4. Geographic concentration (max 15)
    country = node.get("country", "")
    geo_score = COUNTRY_CONCENTRATION.get(country, 5)
    geo_concentration = min(15, geo_score)

    total = single_source_penalty + revenue_exposure + lead_time_risk + geo_concentration

    return {
        "node_id": node.get("id", "unknown"),
        "name": node.get("name", "unknown"),
        "total_score": round(total, 1),
        "breakdown": {
            "single_source_penalty": round(single_source_penalty, 1),
            "revenue_exposure": round(revenue_exposure, 1),
            "lead_time_risk": round(lead_time_risk, 1),
            "geo_concentration": round(geo_concentration, 1),
        },
        "risk_tier": "CRITICAL" if total >= 60 else "HIGH" if total >= 40 else "MEDIUM" if total >= 20 else "LOW",
    }


def score_all_nodes(
    graph_data: dict,
) -> list[dict[str, Any]]:
    """Score all nodes in a supply chain graph.

    Args:
        graph_data: Parsed graph JSON with "nodes" list.

    Returns:
        Sorted list of node scores (highest dependency first).
    """
    nodes = graph_data.get("nodes", [])
    total_revenue = sum(n.get("annual_spend", 0) for n in nodes)

    scores = [dependency_score(n, total_revenue) for n in nodes]
    scores.sort(key=lambda s: s["total_score"], reverse=True)
    return scores
