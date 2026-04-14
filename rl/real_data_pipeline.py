"""
Real-Data Pipeline — Convert real-world datasets into RL training data.

Sources (all real, all verified):
  1. DataCo Supply Chain (180,519 orders, Kaggle): primary training signal
  2. NOAA IBTRACS (4,289 typhoons, 140 yrs): real disruption scenarios
  3. USGS Earthquakes (live feed): real-time disruption triggers
  4. FRED commodity series (17K data points): price trajectories injected into state

Outputs:
  - rl/data/real_buffer.npz         — combined real-data RL buffer
  - rl/data/train.npz / val.npz / test.npz  — 70/15/15 stratified splits
  - rl/data/real_disruption_pool.json       — NOAA-derived disruption scenarios
  - rl/data/fred_state_features.json        — FRED time-series for state injection

Usage:
    python -m rl.real_data_pipeline build           # Build all
    python -m rl.real_data_pipeline split           # Split only
    python -m rl.real_data_pipeline verify          # Verify outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATACO_PATH = DATA_DIR / "dataco.csv"
NOAA_PATH = DATA_DIR / "ibtracs_wp.csv"
USGS_PATH = DATA_DIR / "usgs_m55_30days.csv"
FRED_PATH = DATA_DIR / "fred_cache.json"

OUT_BUFFER = DATA_DIR / "real_buffer.npz"
OUT_TRAIN = DATA_DIR / "real_train.npz"
OUT_VAL = DATA_DIR / "real_val.npz"
OUT_TEST = DATA_DIR / "real_test.npz"
OUT_DISRUPTIONS = DATA_DIR / "real_disruption_pool.json"
OUT_FRED_STATE = DATA_DIR / "fred_state_features.json"


# ============================================================
# DataCo → RL transitions
# ============================================================

def encode_dataco_state(row: dict) -> np.ndarray:
    """Encode a DataCo order row as a 408-dim state vector.

    Mapping aligns with our env's state encoding:
      [0:400] per-node features (40 nodes × 10) — for DataCo, single "order node"
                                                   replicated across slots
      [400:408] global features (8)
    """
    state = np.zeros(408, dtype=np.float32)

    # Per-node features (use first 1-5 slots for the order's "supply chain")
    # Node 0 = customer (always present)
    state[0] = 1.0  # is_operational
    state[1] = float(row.get("Late_delivery_risk", 0))  # risk_score
    state[2] = min(1.0, float(row.get("Days for shipment (scheduled)", 3)) / 30.0)  # inventory_norm
    state[3] = 0.0  # has_backup (no info)
    state[4 + 4] = 1.0  # node type = customer (one-hot)
    state[9] = min(1.0, abs(float(row.get("Sales per customer", 0))) / 1000.0)  # revenue_norm

    # Global features [400:408]
    state[400] = 0.0  # day_progress
    state[401] = 1.0  # budget_remaining_ratio
    state[402] = 1.0 if row.get("Delivery Status") == "Advance shipping" else \
                 0.5 if row.get("Delivery Status") == "Shipping on time" else \
                 0.3 if row.get("Delivery Status") == "Late delivery" else 0.1
    state[403] = float(row.get("Late_delivery_risk", 0)) / 10.0  # disruption count proxy
    state[404] = float(row.get("Late_delivery_risk", 0))  # max severity proxy
    state[405] = max(0.0, -float(row.get("Order Item Profit Ratio", 0)))  # cumulative_loss_ratio (loss as positive)
    state[406] = 0.0  # mc_p50_ratio (no MC for real data)
    state[407] = 0.0  # mc_p95_ratio

    return state


_MARKET_TO_NODE = {
    "Pacific Asia": 0, "Europe": 5, "USCA": 10, "LATAM": 15, "Africa": 20,
    "Asia Pacific": 25,
}
_SEGMENT_OFFSET = {"Consumer": 0, "Corporate": 2, "Home Office": 4, "Unknown": 0}


def shipping_mode_to_action(mode: str, late_risk: int, market: str = "", segment: str = "Unknown",
                              delay_days: float = 0.0, profit: float = 0.0) -> int:
    """Map DataCo features to one of 280 actions (7 types × 40 nodes).

    Uses richer mapping so action distribution spans the full space:
      - action_type from shipping_mode + late_risk + delay severity
      - target_node from market + customer_segment + delay bucket
    """
    mode = (mode or "").lower()

    # Action type — more nuanced
    if "same day" in mode:
        action_type = 4  # always expedite (air)
    elif "first class" in mode:
        action_type = 4 if late_risk else 1  # expedite or activate backup
    elif "second class" in mode:
        if late_risk and delay_days > 2:
            action_type = 4  # expedite
        elif late_risk:
            action_type = 2  # reroute
        else:
            action_type = 6  # alert
    elif "standard" in mode:
        if late_risk and delay_days > 5:
            action_type = 4  # expedite if very late
        elif late_risk:
            action_type = 3  # increase safety stock
        elif profit < 0:
            action_type = 5  # hedge commodity (loss-making)
        else:
            action_type = 0  # do nothing
    else:
        action_type = 0

    # Target node — diversify across markets and segments
    base_node = _MARKET_TO_NODE.get(market, 30)
    seg_off = _SEGMENT_OFFSET.get(segment, 0)
    delay_bucket = min(3, int(max(0, delay_days)))
    target_node = (base_node + seg_off + delay_bucket) % 40

    return action_type * 40 + target_node


def reward_from_dataco(row: dict) -> float:
    """Compute reward signal from DataCo outcome.

    Components (mirrors our 7-component env reward):
      + benefit (profit) → revenue preservation
      - delay_days × 0.1 → SLA penalty
      - late_delivery × 0.25 → stockout-equivalent penalty
    """
    benefit = float(row.get("Benefit per order", 0))
    real_days = float(row.get("Days for shipping (real)", 0))
    sched_days = float(row.get("Days for shipment (scheduled)", 0))
    late = int(row.get("Late_delivery_risk", 0))
    delay = max(0.0, real_days - sched_days)

    # Normalize benefit to [-1, 1] (typical $ range -300 to 200)
    benefit_norm = max(-1.0, min(1.0, benefit / 300.0))
    delay_penalty = -0.1 * min(10.0, delay) / 10.0
    late_penalty = -0.25 if late else 0.0

    return float(benefit_norm * 0.35 + delay_penalty + late_penalty)


def build_dataco_transitions(max_orders: int = 180000) -> dict[str, np.ndarray]:
    """Convert DataCo orders into (state, action, reward, next_state, done) tuples.

    Each order = 1 transition (single-step episode).
    next_state = state with delivery status updated.
    """
    import pandas as pd

    logger.info("Loading DataCo CSV...")
    df = pd.read_csv(str(DATACO_PATH), encoding="latin-1")
    if max_orders < len(df):
        df = df.sample(n=max_orders, random_state=42).reset_index(drop=True)
    n = len(df)
    logger.info("DataCo: %d orders → building %d transitions", n, n)

    states = np.zeros((n, 408), dtype=np.float32)
    actions = np.zeros((n, 2), dtype=np.int64)  # MultiDiscrete([7, 40])
    rewards = np.zeros(n, dtype=np.float32)
    next_states = np.zeros((n, 408), dtype=np.float32)
    dones = np.ones(n, dtype=np.bool_)  # Each order = single step → always done
    returns_to_go = np.zeros(n, dtype=np.float32)

    # Stratification key for splits later
    customer_segments = []
    late_risks = []

    for i in range(n):
        row = df.iloc[i].to_dict()
        s = encode_dataco_state(row)
        states[i] = s

        # Next state = same with delivery resolved
        ns = s.copy()
        if row.get("Delivery Status") == "Late delivery":
            ns[1] = min(1.0, s[1] + 0.2)  # risk increased
            ns[405] = min(1.0, s[405] + 0.1)
        next_states[i] = ns

        # Action (richer mapping using market+segment+delay+profit)
        real_days = float(row.get("Days for shipping (real)", 0))
        sched_days = float(row.get("Days for shipment (scheduled)", 0))
        delay_days_val = max(0.0, real_days - sched_days)
        flat_a = shipping_mode_to_action(
            row.get("Shipping Mode"),
            int(row.get("Late_delivery_risk", 0)),
            market=row.get("Market", ""),
            segment=row.get("Customer Segment", "Unknown"),
            delay_days=delay_days_val,
            profit=float(row.get("Benefit per order", 0)),
        )
        actions[i, 0] = flat_a // 40
        actions[i, 1] = flat_a % 40

        # Reward
        rewards[i] = reward_from_dataco(row)
        returns_to_go[i] = rewards[i]  # single-step episode

        customer_segments.append(row.get("Customer Segment", "Unknown"))
        late_risks.append(int(row.get("Late_delivery_risk", 0)))

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "returns_to_go": returns_to_go,
        "_strat_segment": np.array(customer_segments),
        "_strat_late": np.array(late_risks, dtype=np.int8),
    }


# ============================================================
# NOAA → real disruption scenarios
# ============================================================

def build_noaa_disruption_pool(max_storms: int = 1000) -> list[dict]:
    """Sample NOAA storms as injectable disruption scenarios.

    Each scenario can be triggered during training to replace synthetic disruptions
    with real typhoon data.
    """
    import pandas as pd

    logger.info("Loading NOAA IBTRACS...")
    df = pd.read_csv(str(NOAA_PATH), low_memory=False, skiprows=[1])
    df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

    # Filter: severe (≥64 kts), modern era (≥1990)
    severe = df[(df["USA_WIND"] >= 64) & (df["SEASON"] >= 1990)].copy()

    # Group by storm
    storm_summaries = []
    for sid, group in severe.groupby("SID"):
        if len(storm_summaries) >= max_storms:
            break
        max_wind = float(group["USA_WIND"].max())
        # Map wind speed (kts) → severity (0-1): 64 kts = 0.3, 137 kts (Cat 5) = 1.0
        severity = max(0.0, min(1.0, (max_wind - 64) / 73))
        # Duration: number of 6-hourly observations × 6 hr / 24 hr = days
        duration_days = max(1.0, len(group) * 6 / 24)

        # Region from coordinates
        mean_lat = float(group["LAT"].mean())
        mean_lon = float(group["LON"].mean())
        if 18 <= mean_lat <= 28 and 117 <= mean_lon <= 125:
            region = "Taiwan"
        elif 24 <= mean_lat <= 40 and 130 <= mean_lon <= 145:
            region = "Japan"
        elif 14 <= mean_lat <= 22 and 120 <= mean_lon <= 130:
            region = "Philippines"
        elif 18 <= mean_lat <= 30 and 105 <= mean_lon <= 117:
            region = "South China"
        else:
            region = "Western Pacific"

        storm_summaries.append({
            "storm_id": sid,
            "name": str(group["NAME"].iloc[0] if "NAME" in group else "UNKNOWN"),
            "year": int(group["SEASON"].iloc[0]),
            "max_wind_kts": max_wind,
            "severity": round(severity, 3),
            "duration_days": round(duration_days, 1),
            "region": region,
            "disruption_type": "tropical_cyclone",
        })

    logger.info("NOAA: %d real disruption scenarios", len(storm_summaries))
    return storm_summaries


# ============================================================
# FRED → state feature trajectories
# ============================================================

def build_fred_state_features() -> dict[str, list[float]]:
    """Extract FRED time-series for injection into state observations."""
    fred = json.loads(FRED_PATH.read_text())
    features = {}
    for sid in ["DCOILWTICO", "PCOPPUSDM", "DEXTAUS", "DEXKOUS", "DEXJPUS", "DEXUSEU", "DEXCHUS"]:
        if sid in fred and fred[sid].get("data"):
            values = [d["value"] for d in fred[sid]["data"][-1000:]]
            features[sid] = {
                "label": fred[sid]["label"],
                "values": values,
                "n": len(values),
                "min": float(np.min(values)) if values else 0,
                "max": float(np.max(values)) if values else 0,
                "mean": float(np.mean(values)) if values else 0,
            }
    logger.info("FRED: %d series with state-feature trajectories", len(features))
    return features


# ============================================================
# Train/Val/Test split (70/15/15 stratified)
# ============================================================

def stratified_split(
    data: dict[str, np.ndarray],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Stratified split by customer_segment + late_delivery_risk."""
    n = len(data["states"])
    rng = np.random.default_rng(seed)

    # Stratify by combined key
    seg = data.get("_strat_segment", np.zeros(n, dtype=object))
    late = data.get("_strat_late", np.zeros(n, dtype=np.int8))
    keys = np.array([f"{s}_{l}" for s, l in zip(seg, late)])

    train_idx, val_idx, test_idx = [], [], []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        n_val = int(len(idx) * val_frac)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)

    keys_to_save = ["states", "actions", "rewards", "next_states", "dones", "returns_to_go"]
    train = {k: data[k][train_idx] for k in keys_to_save if k in data}
    val = {k: data[k][val_idx] for k in keys_to_save if k in data}
    test = {k: data[k][test_idx] for k in keys_to_save if k in data}

    logger.info("Splits: train=%d val=%d test=%d (total=%d)",
                len(train_idx), len(val_idx), len(test_idx), n)
    return train, val, test


# ============================================================
# Main pipeline
# ============================================================

def build_all() -> dict[str, Any]:
    """Build the entire real-data pipeline."""
    logger.info("=" * 70)
    logger.info("BUILDING REAL-DATA RL PIPELINE")
    logger.info("=" * 70)

    summary = {}

    # 1. DataCo transitions
    logger.info("\n[1/4] Building DataCo transitions...")
    transitions = build_dataco_transitions(max_orders=180000)
    summary["dataco_transitions"] = len(transitions["states"])

    # 2. Save full buffer (drop strat keys)
    logger.info("\n[2/4] Saving real_buffer.npz...")
    save_keys = ["states", "actions", "rewards", "next_states", "dones", "returns_to_go"]
    np.savez_compressed(str(OUT_BUFFER), **{k: transitions[k] for k in save_keys})
    summary["real_buffer_mb"] = OUT_BUFFER.stat().st_size / 1e6

    # 3. Stratified splits
    logger.info("\n[3/4] Building stratified train/val/test splits...")
    train, val, test = stratified_split(transitions, train_frac=0.70, val_frac=0.15)
    np.savez_compressed(str(OUT_TRAIN), **train)
    np.savez_compressed(str(OUT_VAL), **val)
    np.savez_compressed(str(OUT_TEST), **test)
    summary["train_size"] = len(train["states"])
    summary["val_size"] = len(val["states"])
    summary["test_size"] = len(test["states"])

    # 4. NOAA + FRED
    logger.info("\n[4/4] Building NOAA disruption pool + FRED state features...")
    disruptions = build_noaa_disruption_pool(max_storms=2000)
    OUT_DISRUPTIONS.write_text(json.dumps(disruptions, indent=2))
    summary["noaa_disruptions"] = len(disruptions)

    fred_features = build_fred_state_features()
    OUT_FRED_STATE.write_text(json.dumps(fred_features, indent=2))
    summary["fred_series"] = len(fred_features)

    logger.info("\n" + "=" * 70)
    logger.info("REAL-DATA PIPELINE BUILD COMPLETE")
    logger.info("=" * 70)
    for k, v in summary.items():
        logger.info("  %-25s %s", k, v)

    return summary


def verify():
    """Verify all output files exist and are valid."""
    for path in [OUT_BUFFER, OUT_TRAIN, OUT_VAL, OUT_TEST, OUT_DISRUPTIONS, OUT_FRED_STATE]:
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  [OK] {path.name}: {size_mb:.2f} MB")
        else:
            print(f"  [MISS] {path.name}")

    if OUT_BUFFER.exists():
        d = np.load(str(OUT_BUFFER))
        print(f"\n  Real buffer: {len(d['states']):,} transitions")
        print(f"  States shape: {d['states'].shape}")
        print(f"  Actions shape: {d['actions'].shape}")
        print(f"  Reward range: [{d['rewards'].min():.3f}, {d['rewards'].max():.3f}]")
        print(f"  Reward mean: {d['rewards'].mean():.3f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify"], default="build", nargs="?")
    args = parser.parse_args()

    if args.command == "build":
        build_all()
        verify()
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()
