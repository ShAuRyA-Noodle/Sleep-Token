"""
SupplyMind Fast Engine — Isolated shadow implementation.

This folder contains a Numba/C++-accelerated drop-in replacement for the
most expensive parts of the core environment, WITHOUT modifying any
core files.

Usage:
    from rl.fast_engine import FastMonteCarloEngine
    # Numerically equivalent to server.engine.monte_carlo.MonteCarloEngine
    # but 10-50x faster via Numba JIT compilation

If fast engine fails to load, falls back transparently to the Python engine.
Core 154 tests remain unaffected.
"""

from rl.fast_engine.fast_monte_carlo import FastMonteCarloEngine, FAST_MC_AVAILABLE

__all__ = ["FastMonteCarloEngine", "FAST_MC_AVAILABLE"]
