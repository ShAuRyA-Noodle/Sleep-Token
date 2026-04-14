"""
IQL agent — redirects to pure PyTorch implementation in baselines.py.
No d3rlpy dependency.

Usage:
    python -m rl.offline.iql_agent --steps 100000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl.offline.baselines import train_iql


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train IQL (Pure PyTorch)")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_iql(n_steps=args.steps, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
