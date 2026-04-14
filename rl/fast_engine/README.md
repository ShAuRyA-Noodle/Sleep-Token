# Fast Engine — Numba Shadow Monte Carlo

This folder contains a **drop-in replacement** for `server.engine.monte_carlo.MonteCarloEngine`,
accelerated via Numba JIT compilation.

## Why?

The core env's Monte Carlo (1000 simulations per `env.step()`) is the biggest CPU bottleneck
in RL training — it's ~60-70% of total env stepping time. This shadow implementation runs
the same math in native machine code via Numba JIT.

## Design Rules (keeps core safe)

1. **Zero modifications** to `server/engine/*.py` or any core file
2. **Same interface** as original `MonteCarloEngine.run_simulation()`
3. **Same return shape** (p50_loss, p95_loss, p99_loss, avg_nodes_affected, max_delay_days)
4. **Graceful fallback**: if Numba compile fails, `FAST_MC_AVAILABLE = False` and calling
   code falls back to original Python engine
5. **Training only**: evaluation still uses the original engine for exact grader compatibility

## Files

- `fast_monte_carlo.py` — The Numba-accelerated `FastMonteCarloEngine` class
- `benchmark.py` — Speed + output comparison vs. original Python engine
- `__init__.py` — Clean public exports

## Usage

```python
from rl.fast_engine import FastMonteCarloEngine, FAST_MC_AVAILABLE

if FAST_MC_AVAILABLE:
    engine = FastMonteCarloEngine(seed=42)
else:
    from server.engine.monte_carlo import MonteCarloEngine
    engine = MonteCarloEngine(seed=42)

result = engine.run_simulation(graph, active_disruptions, n_simulations=1000)
# Same result dict either way
```

## Benchmarking

```
python -m rl.fast_engine.benchmark
```

Expected speedup: 10-50x on the MC hot loop.

## Fallback Strategy

If this folder is deleted or imports fail, the main codebase is unaffected —
training/eval simply uses the original Python engine everywhere.
