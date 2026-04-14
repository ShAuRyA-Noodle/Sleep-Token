"""
Temporal Graph Network (TGN) for SupplyMind.

Learns trajectory-based node representations that evolve over time,
not just point-in-time snapshots. Produces per-node 5-day risk trajectories.

Architecture:
    TGNMemory (per-node memory, updated at each timestep) + TransformerConv
    memory_dim=64, time_dim=8, 2 attention heads.

Must call memory.reset_state() at episode start.
~2x slower to train than static GNN.

Only instantiate if PyG >= 2.3 is available.

Usage:
    python -m rl.gnn.tgn --train
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Check PyG availability
_HAS_TGN = False
try:
    import torch_geometric
    from torch_geometric.nn import TransformerConv
    _v = tuple(int(x) for x in torch_geometric.__version__.split(".")[:2])
    if _v >= (2, 3):
        _HAS_TGN = True
        logger.info("PyG >= 2.3 available, TGN enabled")
    else:
        logger.info("PyG version %s < 2.3, TGN disabled", torch_geometric.__version__)
except ImportError:
    logger.info("PyG not installed, TGN disabled")


class SupplyChainTGN(nn.Module):
    """Temporal Graph Network for supply chain dynamics.

    Maintains per-node memory that accumulates temporal information.
    At each timestep, updates memory based on incoming events and
    propagates through graph structure.

    Args:
        node_features:  Input feature dimension per node (10).
        memory_dim:     Per-node memory dimension (64).
        time_dim:       Time encoding dimension (8).
        hidden_dim:     Message passing hidden dimension (64).
        n_heads:        Attention heads (2).
        n_nodes:        Maximum number of nodes (40).
    """

    def __init__(
        self,
        node_features: int = 10,
        memory_dim: int = 64,
        time_dim: int = 8,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_nodes: int = 40,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.n_nodes = n_nodes

        # Per-node memory (not a parameter — updated in-place)
        self.register_buffer("memory", torch.zeros(n_nodes, memory_dim))

        # Time encoding
        self.time_encoder = nn.Linear(1, time_dim)

        # Memory updater: GRU
        self.memory_updater = nn.GRUCell(
            node_features + time_dim, memory_dim,
        )

        # Message passing
        if _HAS_TGN:
            self.conv = TransformerConv(
                memory_dim, hidden_dim, heads=n_heads, concat=False,
            )
        else:
            self.conv = nn.Sequential(
                nn.Linear(memory_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )

        # Risk trajectory predictor: memory -> 5-day risk forecast
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),  # 5-day forecast
            nn.Sigmoid(),
        )

    def reset_memory(self) -> None:
        """Reset per-node memory at episode start."""
        self.memory.zero_()

    def update_memory(
        self,
        node_features: torch.Tensor,
        timestep: float,
    ) -> None:
        """Update per-node memory with new observations.

        Args:
            node_features: (n_nodes, node_features) current observations.
            timestep:      Current time step (float).
        """
        n = min(node_features.shape[0], self.n_nodes)
        time_enc = self.time_encoder(
            torch.full((n, 1), timestep, device=node_features.device)
        )
        update_input = torch.cat([node_features[:n], time_enc], dim=-1)
        self.memory[:n] = self.memory_updater(update_input, self.memory[:n])

    def forward(
        self,
        edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings and risk trajectories.

        Args:
            edge_index: (2, n_edges) edge connectivity.

        Returns:
            embeddings:       (n_nodes, hidden_dim)
            risk_trajectories: (n_nodes, 5) — 5-day risk forecast per node.
        """
        if _HAS_TGN and edge_index is not None:
            embeddings = self.conv(self.memory, edge_index)
        else:
            embeddings = self.conv(self.memory)

        risk_traj = self.trajectory_head(embeddings)
        return embeddings, risk_traj

    def predict_risk_trajectories(
        self,
        node_features: torch.Tensor,
        timestep: float,
        edge_index: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Full pipeline: update memory + predict 5-day risk trajectories.

        Args:
            node_features: (n_nodes, 10)
            timestep:      Current step.
            edge_index:    Graph edges.

        Returns:
            (n_nodes, 5) numpy array of risk scores per day ahead.
        """
        self.eval()
        with torch.no_grad():
            self.update_memory(node_features, timestep)
            _, risk_traj = self.forward(edge_index)
        return risk_traj[:node_features.shape[0]].cpu().numpy()
