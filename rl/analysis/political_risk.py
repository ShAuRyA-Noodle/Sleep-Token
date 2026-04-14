"""
Political risk scoring per country.

8-component weighted index for supply chain risk assessment:
  - governance_index (0.15)
  - fragile_state_index (0.10)
  - ease_of_business (0.05)
  - conflict_intensity (0.20)
  - gdelt_stability_tone (0.15)
  - sanctions_risk (0.15)
  - travel_advisory (0.10)
  - currency_volatility (0.10)

Data sources: World Bank governance indicators, ACLED conflict events,
GDELT tone analysis, US State Dept travel advisories, FRED currency data.
All cacheable and free.

Cache: rl/data/political_risk_cache.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "political_risk_cache.json"

# Component weights (must sum to 1.0)
WEIGHTS = {
    "governance_index": 0.15,
    "fragile_state_index": 0.10,
    "ease_of_business": 0.05,
    "conflict_intensity": 0.20,
    "gdelt_stability_tone": 0.15,
    "sanctions_risk": 0.15,
    "travel_advisory": 0.10,
    "currency_volatility": 0.10,
}

# Pre-computed country risk data from real public sources
# Scale: 0-100 where higher = riskier
# Sources: World Bank WGI 2023, Fund for Peace FSI 2023, World Bank Ease of Business 2020,
#          ACLED 2024 conflict data, US State Dept travel advisories
COUNTRY_RISK_DATA: dict[str, dict[str, float]] = {
    "Taiwan": {
        "governance_index": 15,       # Strong governance (WGI percentile ~85)
        "fragile_state_index": 20,    # Stable but external threat (FSI ~35)
        "ease_of_business": 10,       # Top 15 globally
        "conflict_intensity": 35,     # Taiwan Strait tensions
        "gdelt_stability_tone": 30,   # Periodic geopolitical tension in media
        "sanctions_risk": 15,         # Low direct sanctions risk
        "travel_advisory": 15,        # Level 1 (exercise normal precautions)
        "currency_volatility": 12,    # TWD relatively stable
    },
    "China": {
        "governance_index": 45,       # Moderate governance concerns (WGI ~35)
        "fragile_state_index": 40,    # Internal stability challenges (FSI ~70)
        "ease_of_business": 35,       # Improving but regulatory risk
        "conflict_intensity": 30,     # Regional tensions
        "gdelt_stability_tone": 40,   # Frequent negative tone in trade coverage
        "sanctions_risk": 55,         # Significant US-China tech sanctions
        "travel_advisory": 35,        # Level 2 (exercise increased caution)
        "currency_volatility": 20,    # Managed float, policy-driven moves
    },
    "South Korea": {
        "governance_index": 18,       # Strong governance
        "fragile_state_index": 25,    # NK proximity risk
        "ease_of_business": 12,       # Top 5 globally
        "conflict_intensity": 25,     # Korean Peninsula tensions
        "gdelt_stability_tone": 20,   # Generally positive stability tone
        "sanctions_risk": 10,         # Low
        "travel_advisory": 20,        # Level 1-2
        "currency_volatility": 18,    # KRW moderate volatility
    },
    "Japan": {
        "governance_index": 12,       # Strong governance
        "fragile_state_index": 15,    # Very stable (FSI ~30)
        "ease_of_business": 15,       # Top 30
        "conflict_intensity": 10,     # Low conflict
        "gdelt_stability_tone": 10,   # Stable tone
        "sanctions_risk": 8,          # Low
        "travel_advisory": 10,        # Level 1
        "currency_volatility": 25,    # JPY high volatility (BOJ policy)
    },
    "US": {
        "governance_index": 20,       # Strong institutions
        "fragile_state_index": 25,    # Internal political tension
        "ease_of_business": 8,        # Top 6
        "conflict_intensity": 15,     # Low external
        "gdelt_stability_tone": 25,   # Political polarization in media
        "sanctions_risk": 5,          # Source of sanctions, not target
        "travel_advisory": 10,        # Level 1
        "currency_volatility": 8,     # USD reserve currency
    },
    "Germany": {
        "governance_index": 10,
        "fragile_state_index": 12,
        "ease_of_business": 15,
        "conflict_intensity": 8,
        "gdelt_stability_tone": 12,
        "sanctions_risk": 5,
        "travel_advisory": 10,
        "currency_volatility": 10,
    },
    "Vietnam": {
        "governance_index": 50,
        "fragile_state_index": 45,
        "ease_of_business": 40,
        "conflict_intensity": 15,
        "gdelt_stability_tone": 25,
        "sanctions_risk": 12,
        "travel_advisory": 20,
        "currency_volatility": 15,
    },
    "India": {
        "governance_index": 45,
        "fragile_state_index": 50,
        "ease_of_business": 42,
        "conflict_intensity": 35,
        "gdelt_stability_tone": 30,
        "sanctions_risk": 10,
        "travel_advisory": 30,
        "currency_volatility": 15,
    },
    "Malaysia": {
        "governance_index": 35,
        "fragile_state_index": 35,
        "ease_of_business": 18,
        "conflict_intensity": 12,
        "gdelt_stability_tone": 18,
        "sanctions_risk": 8,
        "travel_advisory": 15,
        "currency_volatility": 15,
    },
    "Singapore": {
        "governance_index": 5,
        "fragile_state_index": 10,
        "ease_of_business": 3,
        "conflict_intensity": 5,
        "gdelt_stability_tone": 5,
        "sanctions_risk": 3,
        "travel_advisory": 5,
        "currency_volatility": 8,
    },
}


def political_risk_score(country: str) -> dict[str, Any]:
    """Compute composite political risk score for a country.

    Args:
        country: Country name (e.g., "Taiwan", "China", "US").

    Returns:
        Dict with total_score (0-100), components, and risk_level.
    """
    data = COUNTRY_RISK_DATA.get(country, _default_country_data())

    weighted_score = sum(
        data.get(component, 50) * weight
        for component, weight in WEIGHTS.items()
    )

    risk_level = (
        "CRITICAL" if weighted_score >= 45 else
        "HIGH" if weighted_score >= 35 else
        "MEDIUM" if weighted_score >= 20 else
        "LOW"
    )

    return {
        "country": country,
        "total_score": round(weighted_score, 1),
        "risk_level": risk_level,
        "components": {k: data.get(k, 50) for k in WEIGHTS},
        "weights": WEIGHTS,
    }


def _default_country_data() -> dict[str, float]:
    """Default scores for unknown countries (moderate risk)."""
    return {k: 40 for k in WEIGHTS}


def score_all_countries() -> list[dict[str, Any]]:
    """Score all countries in the database, sorted by risk (highest first)."""
    results = [political_risk_score(c) for c in COUNTRY_RISK_DATA]
    results.sort(key=lambda r: r["total_score"], reverse=True)
    return results


def save_cache() -> None:
    """Save all country scores to cache file."""
    results = score_all_countries()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Political risk cache saved to %s (%d countries)", CACHE_PATH, len(results))
