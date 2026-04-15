"""
Disruption confidence scoring for supply chain risk predictions.

Multi-signal corroboration scoring for 72-hour disruption predictions:
  - prediction_probability * 0.5
  - historical_accuracy * 0.3
  - corroboration_bonus (min 0.2, indicator_count * 0.05)

Thresholds:
  >= 0.8 -> RED ALERT (immediate notification, auto-draft actions)
  >= 0.5 -> AMBER WARNING (dashboard highlight, daily digest)
  >= 0.3 -> YELLOW WATCH (monitor, weekly report)
  <  0.3 -> GREEN (no action needed)
"""

from __future__ import annotations

from typing import Any


def disruption_confidence(
    prediction_probability: float,
    indicator_count: int,
    historical_accuracy: float = 0.7,
) -> dict[str, Any]:
    """Compute composite confidence for a 72-hour disruption prediction.

    Args:
        prediction_probability: Model's estimated probability of disruption (0-1).
        indicator_count:        Number of corroborating early warning signals.
        historical_accuracy:    Past accuracy of similar predictions (0-1).

    Returns:
        Dict with confidence score, alert level, and component breakdown.
    """
    corroboration_bonus = min(0.2, indicator_count * 0.05)

    raw_confidence = (
        prediction_probability * 0.5
        + historical_accuracy * 0.3
        + corroboration_bonus
    )

    confidence = min(1.0, max(0.0, raw_confidence))

    # Determine alert level
    if confidence >= 0.8:
        alert_level = "RED"
        alert_description = "ALERT: Immediate notification, auto-draft mitigation actions"
    elif confidence >= 0.5:
        alert_level = "AMBER"
        alert_description = "WARNING: Dashboard highlight, include in daily digest"
    elif confidence >= 0.3:
        alert_level = "YELLOW"
        alert_description = "WATCH: Continue monitoring, include in weekly report"
    else:
        alert_level = "GREEN"
        alert_description = "No immediate action required"

    return {
        "confidence": round(confidence, 3),
        "alert_level": alert_level,
        "alert_description": alert_description,
        "components": {
            "prediction_probability": round(prediction_probability, 3),
            "historical_accuracy": round(historical_accuracy, 3),
            "indicator_count": indicator_count,
            "corroboration_bonus": round(corroboration_bonus, 3),
        },
        "weights": {
            "prediction": 0.5,
            "historical": 0.3,
            "corroboration": "min(0.2, count * 0.05)",
        },
    }


def batch_confidence(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score multiple disruption predictions at once.

    Args:
        predictions: List of dicts with prediction_probability,
                     indicator_count, and optional historical_accuracy.

    Returns:
        List of confidence results, sorted by confidence (highest first).
    """
    results = []
    for pred in predictions:
        result = disruption_confidence(
            prediction_probability=pred.get("prediction_probability", 0.5),
            indicator_count=pred.get("indicator_count", 0),
            historical_accuracy=pred.get("historical_accuracy", 0.7),
        )
        result["disruption_type"] = pred.get("disruption_type", "unknown")
        result["region"] = pred.get("region", "unknown")
        results.append(result)

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results
