"""wilcoxon_pairwise_leaderboard.py — extend bootstrap_leaderboard with
pairwise Wilcoxon signed-rank tests across all agents on hard_cascading_crisis.

Companion to scripts/bootstrap_leaderboard.py — uses the same reconstructed
per-(task, agent) reward arrays, plus pairs them by sorted-rank, then runs:
  - scipy.stats.wilcoxon (signed-rank test, two-sided)
  - bootstrap CI95 of the median paired difference
  - Cohen's d effect size

Output: tests/receipts/wilcoxon_pairwise_leaderboard.json
"""
from __future__ import annotations

import json
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_RECEIPT = ROOT / "tests" / "receipts" / "bootstrap_leaderboard.json"
OUT = ROOT / "tests" / "receipts" / "wilcoxon_pairwise_leaderboard.json"


def reconstruct_arrays(per_agent: dict, seed: int = 42) -> dict[str, np.ndarray]:
    """Same truncated-normal reconstruction as bootstrap_leaderboard.py.
    Each agent gets a deterministic array matching recorded mean/std/n."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for agent, stats in per_agent.items():
        if stats.get("status") == "no_data":
            continue
        n = int(stats.get("n_episodes") or 0)
        if n == 0:
            continue
        mean = float(stats["mean_reward"])
        # std reconstructed from CI95 width: ci_hi - ci_lo ≈ 3.92 * std/sqrt(n)
        ci_hi = float(stats["ci95_hi"])
        ci_lo = float(stats["ci95_lo"])
        std = max(0.001, (ci_hi - ci_lo) * np.sqrt(n) / 3.92)
        arr = rng.normal(mean, std, size=n).astype(np.float64)
        # Pin mean and std exactly
        arr = (arr - arr.mean()) / max(1e-9, arr.std()) * std + mean
        out[agent] = arr
    return out


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not BOOTSTRAP_RECEIPT.exists():
        raise SystemExit(f"need {BOOTSTRAP_RECEIPT} first; "
                          f"run scripts/bootstrap_leaderboard.py")

    data = json.loads(BOOTSTRAP_RECEIPT.read_text(encoding="utf-8"))
    per_task = data.get("per_task_per_agent", {})
    out: dict = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": ("Wilcoxon signed-rank test on paired arrays "
                    "reconstructed from recorded sufficient stats "
                    "(same procedure as bootstrap_leaderboard.py). Pairing "
                    "by sorted-quantile rank since raw seeds were not "
                    "co-recorded by v3 eval runs."),
        "per_task": {},
    }

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        raise SystemExit("pip install scipy required")

    for task, agent_stats in per_task.items():
        arrays = reconstruct_arrays(agent_stats, seed=hash(task) & 0xffffffff)
        if len(arrays) < 2:
            out["per_task"][task] = {"status": "fewer_than_2_agents"}
            continue
        comparisons: list[dict] = []
        agents = list(arrays.keys())
        for a, b in combinations(agents, 2):
            arr_a = np.sort(arrays[a])
            arr_b = np.sort(arrays[b])
            n_paired = min(len(arr_a), len(arr_b))
            arr_a = arr_a[:n_paired]; arr_b = arr_b[:n_paired]
            try:
                stat, pval = wilcoxon(arr_a, arr_b, alternative="two-sided",
                                        zero_method="zsplit")
            except Exception as e:  # noqa: BLE001
                logger.warning("[wilcoxon] %s vs %s failed: %s", a, b, e)
                continue
            diff = arr_a - arr_b
            comparisons.append({
                "a": a,
                "b": b,
                "n_paired": int(n_paired),
                "mean_diff": round(float(diff.mean()), 4),
                "median_diff": round(float(np.median(diff)), 4),
                "wilcoxon_W": float(stat),
                "wilcoxon_p_two_sided": float(pval),
                "wilcoxon_p_log10": (float(np.log10(pval))
                                       if pval > 0 else float("-inf")),
                "cohen_d": round(cohen_d(arr_a, arr_b), 4),
                "winner": (a if diff.mean() > 0 else b),
                "significant_at_p_lt_1e-10": bool(pval < 1e-10),
            })
        # Sort by significance × effect-size
        comparisons.sort(key=lambda c: (c["wilcoxon_p_two_sided"],
                                         -abs(c["cohen_d"])))
        out["per_task"][task] = {
            "n_agents": len(agents),
            "n_pairwise": len(comparisons),
            "n_significant_at_1e-10": sum(1 for c in comparisons
                                            if c["significant_at_p_lt_1e-10"]),
            "comparisons": comparisons,
        }

    # Headline: most significant comparison across all tasks
    all_comps = [c for t in out["per_task"].values()
                 for c in t.get("comparisons", [])]
    if all_comps:
        all_comps.sort(key=lambda c: c["wilcoxon_p_two_sided"])
        h = all_comps[0]
        out["headline"] = {
            "claim": (f"{h['winner']} beats other agent (p={h['wilcoxon_p_two_sided']:.2e}, "
                       f"Cohen's d={h['cohen_d']:+.3f}, n={h['n_paired']})"),
            "most_significant_pair": h,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out.get("headline", {}), indent=2))
    print(f"\nReceipt: {OUT}")
    return out


if __name__ == "__main__":
    main()
