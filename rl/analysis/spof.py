"""
Single Point of Failure (SPOF) detection for supply chain graphs.

For each component, finds all supply paths from suppliers to factories.
Computes the intersection of nodes across all paths. Any node in ALL
paths is a SPOF — its failure disconnects the supply chain.

Sorts by revenue_at_risk descending.

Source: Network reliability theory applied to supply chain topology.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def detect_spofs(graph_path: str | Path) -> list[dict[str, Any]]:
    """Detect single points of failure in a supply chain graph.

    Args:
        graph_path: Path to graph JSON file.

    Returns:
        List of SPOF dicts sorted by revenue_at_risk (descending).
    """
    with open(graph_path) as f:
        data = json.load(f)

    # Build NetworkX DiGraph
    G = nx.DiGraph()
    node_data = {}
    for node in data["nodes"]:
        nid = node["id"]
        G.add_node(nid)
        node_data[nid] = node

    for edge in data.get("edges", []):
        G.add_edge(edge["source"], edge["target"])

    # Find supplier and factory nodes
    suppliers = [n["id"] for n in data["nodes"] if n["node_type"] == "supplier"]
    factories = [n["id"] for n in data["nodes"] if n["node_type"] == "factory"]
    customers = [n["id"] for n in data["nodes"] if n["node_type"] == "customer"]

    # For each (supplier, factory/customer) pair, find all simple paths
    target_nodes = factories + customers
    if not target_nodes:
        target_nodes = [n["id"] for n in data["nodes"] if n["node_type"] != "supplier"]

    spofs = []
    seen_spofs = set()

    for supplier in suppliers:
        for target in target_nodes:
            try:
                all_paths = list(nx.all_simple_paths(G, supplier, target, cutoff=10))
            except nx.NetworkXError:
                continue

            if len(all_paths) == 0:
                continue

            # Find nodes common to ALL paths (excluding source and target)
            path_node_sets = [set(p[1:-1]) for p in all_paths]  # Exclude endpoints
            if not path_node_sets:
                continue

            common = path_node_sets[0]
            for s in path_node_sets[1:]:
                common = common & s

            # Each common node is a SPOF for this supply path
            for node_id in common:
                if node_id in seen_spofs:
                    continue
                seen_spofs.add(node_id)

                nd = node_data.get(node_id, {})
                revenue_at_risk = _compute_revenue_at_risk(node_id, G, node_data, target_nodes)

                spofs.append({
                    "node_id": node_id,
                    "name": nd.get("name", node_id),
                    "node_type": nd.get("node_type", "unknown"),
                    "country": nd.get("country", "unknown"),
                    "revenue_at_risk": revenue_at_risk,
                    "paths_through": len(all_paths),
                    "has_backup": bool(nd.get("backup_supplier_ids")),
                    "mitigation": _suggest_mitigation(nd),
                })

    # Sort by revenue at risk
    spofs.sort(key=lambda s: s["revenue_at_risk"], reverse=True)
    return spofs


def _compute_revenue_at_risk(
    node_id: str,
    G: nx.DiGraph,
    node_data: dict,
    target_nodes: list[str],
) -> float:
    """Estimate total downstream revenue at risk if this node fails."""
    downstream = set()
    try:
        downstream = nx.descendants(G, node_id)
    except nx.NetworkXError:
        pass

    revenue = 0.0
    for nid in downstream:
        nd = node_data.get(nid, {})
        revenue += nd.get("annual_spend", 0)

    # If node itself has annual_spend, include it
    nd = node_data.get(node_id, {})
    revenue += nd.get("annual_spend", 0)

    return revenue


def _suggest_mitigation(node: dict) -> str:
    """Suggest mitigation for a SPOF node."""
    if node.get("node_type") == "port":
        return "CRITICAL: Identify alternate port routes and pre-negotiate rerouting agreements"
    if node.get("node_type") == "supplier":
        if node.get("backup_supplier_ids"):
            return "HIGH: Backup exists but validate qualification and lead times"
        return "CRITICAL: Qualify alternative supplier immediately"
    if node.get("node_type") == "warehouse":
        return "HIGH: Increase safety stock buffer and identify alternate storage"
    return "MEDIUM: Assess redundancy options and develop contingency plan"
