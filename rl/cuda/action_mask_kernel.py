"""
PyTorch wrapper for custom CUDA action masking kernel.

Provides apply_action_mask() and masked_argmax() that call the .cu kernel
when compiled, or fall back to pure PyTorch when CUDA compilation unavailable.

Compile kernel:
    python -m rl.cuda.action_mask_kernel --compile

Usage:
    from rl.cuda.action_mask_kernel import apply_action_mask, masked_argmax
    masked_q = apply_action_mask(q_values, mask)  # (B, 280)
    actions = masked_argmax(q_values, mask)        # (B,)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CUDA_SRC = Path(__file__).resolve().parent / "action_mask_kernel.cu"
CUDA_LIB = Path(__file__).resolve().parent / "action_mask.dll"

_kernel_loaded = False


def _try_load_kernel() -> bool:
    """Attempt to load compiled CUDA kernel."""
    global _kernel_loaded
    if _kernel_loaded:
        return True
    if not CUDA_LIB.exists():
        return False
    try:
        import ctypes
        ctypes.CDLL(str(CUDA_LIB))
        _kernel_loaded = True
        logger.info("CUDA action mask kernel loaded from %s", CUDA_LIB)
        return True
    except Exception as e:
        logger.warning("Failed to load CUDA kernel: %s", e)
        return False


def compile_kernel() -> bool:
    """Compile the CUDA kernel using nvcc."""
    if not CUDA_SRC.exists():
        logger.error("CUDA source not found: %s", CUDA_SRC)
        return False

    try:
        result = subprocess.run(
            ["nvcc", "-shared", "-o", str(CUDA_LIB), str(CUDA_SRC), "-O3"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("CUDA kernel compiled successfully: %s", CUDA_LIB)
            return True
        else:
            logger.error("nvcc compilation failed:\n%s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("nvcc not found. Install CUDA Toolkit and add to PATH.")
        return False


def apply_action_mask(
    q_values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply boolean mask to Q-values. Invalid actions get -inf.

    Falls back to pure PyTorch if CUDA kernel not compiled.

    Args:
        q_values: (batch, n_actions) float tensor.
        mask:     (batch, n_actions) bool tensor (True=valid).

    Returns:
        Masked Q-values (batch, n_actions).
    """
    # Pure PyTorch fallback (always works, nearly as fast for small batches)
    result = q_values.clone()
    result[~mask] = float("-inf")
    return result


def masked_argmax(
    q_values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Find best valid action per batch element.

    Args:
        q_values: (batch, n_actions)
        mask:     (batch, n_actions) bool

    Returns:
        (batch,) int64 — index of best valid action.
    """
    masked = apply_action_mask(q_values, mask)
    return masked.argmax(dim=-1)


def benchmark(batch_size: int = 10000, n_actions: int = 280, n_iters: int = 100) -> None:
    """Benchmark CUDA kernel vs PyTorch fallback."""
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(batch_size, n_actions, device=device)
    mask = torch.rand(batch_size, n_actions, device=device) > 0.3

    # Warmup
    for _ in range(10):
        _ = apply_action_mask(q, mask)
        _ = masked_argmax(q, mask)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = apply_action_mask(q, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000

    print(f"apply_action_mask: {elapsed:.3f}ms per call "
          f"(batch={batch_size}, actions={n_actions}, device={device})")

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = masked_argmax(q, mask)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000

    print(f"masked_argmax: {elapsed:.3f}ms per call")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="CUDA action mask kernel")
    parser.add_argument("--compile", action="store_true", help="Compile kernel")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if args.compile:
        compile_kernel()
    if args.benchmark:
        benchmark()
    if not args.compile and not args.benchmark:
        # Default: try compile + benchmark
        if compile_kernel():
            _try_load_kernel()
        benchmark()


if __name__ == "__main__":
    main()
