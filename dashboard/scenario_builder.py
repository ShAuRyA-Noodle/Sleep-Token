"""
What-If Scenario Builder for SupplyMind Dashboard.

Interactive panel where judges play with the environment directly.

UI Controls:
  - Crisis type dropdown: earthquake, war, pandemic, port_closure, cyber_attack, trade_war, financial_crisis
  - Severity slider: 0.0 -> 1.0
  - Affected region dropdown: Taiwan, China, Europe, US West Coast, Red Sea, Japan
  - Duration slider: 7 -> 90 days
  - [Run Scenario] button

CRISIS_TEMPLATES maps each type to:
  - node_filter: function selecting affected nodes
  - risk_spike: severity-dependent risk increase
  - duration_model: deterministic or stochastic
  - cascade_probability: severity-dependent cascade chance
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


CRISIS_TEMPLATES: dict[str, dict[str, Any]] = {
    "earthquake": {
        "label": "Earthquake",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("country", "").lower() in _region_countries(region)
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.9),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.9, severity * 0.7),
        "description": "Seismic event disrupting manufacturing and infrastructure",
    },
    "war": {
        "label": "Geopolitical Conflict",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("country", "").lower() in _region_countries(region)
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.95),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.95, severity * 0.9),
        "description": "Armed conflict or military action affecting supply routes",
    },
    "pandemic": {
        "label": "Pandemic",
        "node_filter": lambda nodes, region: nodes,  # Global impact
        "risk_spike": lambda severity: min(1.0, severity * 0.6),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(1.0, severity * 0.8),
        "description": "Global health emergency with workforce and logistics disruption",
    },
    "port_closure": {
        "label": "Port Closure",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("node_type") == "port"
            and n.get("country", "").lower() in _region_countries(region)
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.85),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.8, severity * 0.6),
        "description": "Major port shutdown due to weather, labor, or security",
    },
    "cyber_attack": {
        "label": "Cyber Attack",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("node_type") in ("supplier", "factory", "warehouse")
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.7),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.5, severity * 0.5),
        "description": "Ransomware or infrastructure attack on supply chain systems",
    },
    "trade_war": {
        "label": "Trade War / Sanctions",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("country", "").lower() in _region_countries(region)
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.5),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.7, severity * 0.6),
        "description": "Tariffs, export controls, or sanctions affecting trade flows",
    },
    "financial_crisis": {
        "label": "Financial Crisis",
        "node_filter": lambda nodes, region: [
            n for n in nodes if n.get("node_type") == "supplier"
        ],
        "risk_spike": lambda severity: min(1.0, severity * 0.4),
        "duration_model": "deterministic",
        "cascade_probability": lambda severity: min(0.4, severity * 0.3),
        "description": "Supplier bankruptcy or credit crisis affecting procurement",
    },
}

REGION_MAP = {
    "taiwan": ["taiwan"],
    "china": ["china"],
    "europe": ["germany", "netherlands", "france", "uk", "italy"],
    "us west coast": ["us", "usa", "united states"],
    "red sea": ["egypt", "yemen", "saudi arabia", "djibouti"],
    "japan": ["japan"],
    "south korea": ["south korea", "korea"],
    "southeast asia": ["vietnam", "malaysia", "singapore", "indonesia", "thailand"],
}


def _region_countries(region: str) -> list[str]:
    return REGION_MAP.get(region.lower(), [region.lower()])


def render_scenario_builder() -> None:
    """Render the What-If Scenario Builder panel in Streamlit."""
    import streamlit as st

    st.subheader("What-If Scenario Builder")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crisis_type = st.selectbox(
            "Crisis Type",
            list(CRISIS_TEMPLATES.keys()),
            format_func=lambda x: CRISIS_TEMPLATES[x]["label"],
        )
    with col2:
        severity = st.slider("Severity", 0.0, 1.0, 0.7, 0.05)
    with col3:
        region = st.selectbox("Region", [
            "Taiwan", "China", "Europe", "US West Coast",
            "Red Sea", "Japan", "South Korea", "Southeast Asia",
        ])
    with col4:
        duration = st.slider("Duration (days)", 7, 90, 30)

    template = CRISIS_TEMPLATES[crisis_type]
    st.caption(template["description"])

    if st.button("Run Scenario", key="scenario_run"):
        import json

        # Load graph
        task_graph_map = {
            "easy_typhoon_response": "server/data/graphs/easy_graph.json",
            "medium_multi_front": "server/data/graphs/medium_graph.json",
            "hard_cascading_crisis": "server/data/graphs/hard_graph.json",
        }
        graph_path = _PROJECT_ROOT / task_graph_map.get(
            st.session_state.get("task_id", "easy_typhoon_response"),
            "server/data/graphs/easy_graph.json"
        )
        graph_data = json.loads(graph_path.read_text())

        affected_nodes = template["node_filter"](graph_data["nodes"], region)
        risk_spike = template["risk_spike"](severity)
        cascade_prob = template["cascade_probability"](severity)

        # Display results
        st.markdown(f"### Scenario: {template['label']} in {region}")
        st.markdown(f"- **Affected nodes:** {len(affected_nodes)}")
        st.markdown(f"- **Risk spike:** +{risk_spike:.0%}")
        st.markdown(f"- **Duration:** {duration} days")
        st.markdown(f"- **Cascade probability:** {cascade_prob:.0%}")

        if affected_nodes:
            st.markdown("**Affected:**")
            for n in affected_nodes[:10]:
                name = n.get("name", n.get("id", "?"))
                ntype = n.get("node_type", "?")
                st.write(f"  - {name} ({ntype})")
        else:
            st.warning("No nodes in this region match the crisis filter.")

        # Impact estimation
        from rl.analysis.financial_impact import estimate_cascade_impact
        impact = estimate_cascade_impact(affected_nodes, severity, duration)
        st.metric("Estimated Daily Impact", f"${impact['total_daily_impact']:,.0f}")
        st.metric("Total Estimated Impact", f"${impact['total_impact']:,.0f}")
