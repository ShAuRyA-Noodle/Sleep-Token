"""
MC Dropout uncertainty quantification for SupplyMind RL policies.

Keeps model.train() during inference and runs 50 forward passes with
dropout active. The variance across passes = epistemic uncertainty
(model uncertainty about what to do).

Output: mean action values + epistemic uncertainty (std per action).

Usage:
    from rl.uncertainty import mc_dropout_predict
    mean_q, std_q, action = mc_dropout_predict(model, state, n_passes=50)
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    state: np.ndarray | torch.Tensor,
    n_passes: int = 50,
    action_mask: np.ndarray | torch.Tensor | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run MC Dropout inference to quantify epistemic uncertainty.

    The model MUST have dropout layers. We keep model.train() so dropout
    remains active, then run n_passes forward passes on the same input.
    The standard deviation across passes measures model uncertainty.

    Args:
        model:       Neural network with dropout layers (QR-DQN, BC, etc.).
        state:       (state_dim,) observation vector.
        n_passes:    Number of stochastic forward passes (default 50).
        action_mask: (n_actions,) boolean mask, True = valid action.
        device:      Torch device for inference (use CPU for dashboard).

    Returns:
        mean_q:       (n_actions,) mean Q-values across passes.
        std_q:        (n_actions,) std dev (epistemic uncertainty).
        best_action:  int — action with highest mean Q (masked).
    """
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()
    state = state.to(device).unsqueeze(0)  # (1, state_dim)

    # Keep dropout active
    model.train()

    all_outputs = []
    for _ in range(n_passes):
        output = model(state)  # Shape depends on model type

        # Handle QR-DQN (outputs quantile values)
        if output.dim() == 3:
            # (1, n_actions, n_quantiles) -> mean over quantiles
            q_values = output.mean(dim=-1)  # (1, n_actions)
        else:
            # (1, n_actions) direct Q-values or logits
            q_values = output

        all_outputs.append(q_values.squeeze(0).cpu().numpy())

    model.eval()  # Restore eval mode

    all_outputs_np = np.stack(all_outputs, axis=0)  # (n_passes, n_actions)
    mean_q = all_outputs_np.mean(axis=0)
    std_q = all_outputs_np.std(axis=0)

    # Apply mask
    if action_mask is not None:
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.numpy()
        mean_q_masked = mean_q.copy()
        mean_q_masked[~action_mask] = float("-inf")
        best_action = int(np.argmax(mean_q_masked))
    else:
        best_action = int(np.argmax(mean_q))

    return mean_q, std_q, best_action


def get_uncertainty_summary(
    std_q: np.ndarray,
    top_k: int = 5,
) -> dict:
    """Summarize uncertainty for logging/dashboard.

    Args:
        std_q: (n_actions,) epistemic uncertainty per action.
        top_k: Number of most/least uncertain actions to report.

    Returns:
        Dict with overall stats and top uncertain/certain actions.
    """
    return {
        "mean_uncertainty": float(np.mean(std_q)),
        "max_uncertainty": float(np.max(std_q)),
        "min_uncertainty": float(np.min(std_q)),
        "most_uncertain_actions": np.argsort(std_q)[-top_k:][::-1].tolist(),
        "most_certain_actions": np.argsort(std_q)[:top_k].tolist(),
        "high_uncertainty_fraction": float((std_q > np.mean(std_q) + np.std(std_q)).mean()),
    }
