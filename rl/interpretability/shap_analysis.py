"""
SHAP analysis on RL policies for SupplyMind.

Uses shap.DeepExplainer with 100 background states to compute feature
importance for each action decision. Decodes the 408-float state back
to named features for human-readable output.

Returns top 10 most influential features per action.
Computes on CPU for dashboard (GPU for training only).

Usage:
    from rl.interpretability.shap_analysis import explain_with_shap
    top_features = explain_with_shap(model, state, background_states)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Feature names for the 408-float state vector
MAX_NODES = 40
FEATURES_PER_NODE = 10
NODE_FEATURE_NAMES = [
    "is_operational",
    "risk_score",
    "inventory_days_norm",
    "has_backup",
    "type_supplier",
    "type_warehouse",
    "type_port",
    "type_factory",
    "type_customer",
    "revenue_norm",
]
GLOBAL_FEATURE_NAMES = [
    "day_progress",
    "budget_remaining_ratio",
    "health_score_norm",
    "disruption_count_norm",
    "max_severity",
    "cumulative_loss_ratio",
    "mc_p50_ratio",
    "mc_p95_ratio",
]


def get_feature_names(node_ids: list[str] | None = None) -> list[str]:
    """Generate human-readable names for all 408 state features.

    Args:
        node_ids: List of node IDs (e.g., ["SUP_TSMC", "WH_TAIWAN", ...]).
                  If None, uses generic "node_0", "node_1", etc.

    Returns:
        List of 408 feature name strings.
    """
    names = []
    for i in range(MAX_NODES):
        node_label = node_ids[i] if node_ids and i < len(node_ids) else f"node_{i}"
        for feat in NODE_FEATURE_NAMES:
            names.append(f"{node_label}/{feat}")
    for feat in GLOBAL_FEATURE_NAMES:
        names.append(f"global/{feat}")
    return names


def explain_with_shap(
    model: torch.nn.Module,
    state: np.ndarray,
    background_states: np.ndarray,
    node_ids: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute SHAP values for a single state prediction.

    Args:
        model:             PyTorch model (QR-DQN, BC, etc.) on CPU.
        state:             (408,) state to explain.
        background_states: (N, 408) reference states (100 recommended).
        node_ids:          Node ID list for feature naming.
        top_k:             Number of top features to return.

    Returns:
        Dict with:
          - top_positive: list of (feature_name, shap_value) pushing toward action
          - top_negative: list of (feature_name, shap_value) pushing against action
          - predicted_action: int
          - all_shap_values: (408,) SHAP values for the predicted action
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Returning gradient-based fallback.")
        return _gradient_fallback(model, state, node_ids, top_k)

    model.eval()
    model.cpu()

    # Wrapper to output Q-values (mean over quantiles if QR-DQN)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if out.dim() == 3:
                return out.mean(dim=-1)
            return out

    wrapper = ModelWrapper(model)

    bg = torch.from_numpy(background_states[:100]).float()
    state_t = torch.from_numpy(state).float().unsqueeze(0)

    try:
        explainer = shap.DeepExplainer(wrapper, bg)
        shap_values = explainer.shap_values(state_t)
    except Exception as e:
        logger.warning("SHAP DeepExplainer failed: %s. Using GradientExplainer.", e)
        try:
            explainer = shap.GradientExplainer(wrapper, bg)
            shap_values = explainer.shap_values(state_t)
        except Exception as e2:
            logger.warning("GradientExplainer also failed: %s. Returning gradient fallback.", e2)
            return _gradient_fallback(model, state, node_ids, top_k)

    # Get predicted action
    with torch.no_grad():
        q_values = wrapper(state_t).numpy()[0]
    predicted_action = int(np.argmax(q_values))

    # SHAP values for the predicted action
    if isinstance(shap_values, list):
        sv = shap_values[predicted_action][0]  # (408,)
    else:
        sv = shap_values[0]  # (408,)

    feature_names = get_feature_names(node_ids)

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sv))[::-1]

    top_positive = []
    top_negative = []
    for idx in sorted_idx:
        name = feature_names[idx]
        val = float(sv[idx])
        if val > 0 and len(top_positive) < top_k:
            top_positive.append((name, round(val, 6)))
        elif val < 0 and len(top_negative) < top_k:
            top_negative.append((name, round(val, 6)))
        if len(top_positive) >= top_k and len(top_negative) >= top_k:
            break

    return {
        "predicted_action": predicted_action,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "all_shap_values": sv.tolist(),
        "feature_names": feature_names,
    }


def _gradient_fallback(
    model: torch.nn.Module,
    state: np.ndarray,
    node_ids: list[str] | None,
    top_k: int,
) -> dict[str, Any]:
    """Gradient-based feature importance when SHAP is unavailable."""
    model.eval()
    state_t = torch.from_numpy(state).float().unsqueeze(0).requires_grad_(True)

    output = model(state_t)
    if output.dim() == 3:
        q_values = output.mean(dim=-1)
    else:
        q_values = output

    predicted_action = int(q_values.argmax(dim=-1).item())
    q_values[0, predicted_action].backward()

    grads = state_t.grad[0].detach().numpy()
    feature_names = get_feature_names(node_ids)

    sorted_idx = np.argsort(np.abs(grads))[::-1]
    top_positive = [(feature_names[i], round(float(grads[i]), 6))
                    for i in sorted_idx[:top_k] if grads[i] > 0]
    top_negative = [(feature_names[i], round(float(grads[i]), 6))
                    for i in sorted_idx[:top_k] if grads[i] < 0]

    return {
        "predicted_action": predicted_action,
        "top_positive": top_positive[:top_k],
        "top_negative": top_negative[:top_k],
        "all_shap_values": grads.tolist(),
        "feature_names": feature_names,
        "method": "gradient_fallback",
    }
