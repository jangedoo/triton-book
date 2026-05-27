"""Reusable benchmark helpers.

All helpers take a zero-arg callable and return time in milliseconds. They
use triton.testing.do_bench, which handles warmup, cuda sync, and a sensible
default of median reduction.
"""

from typing import Callable, Dict, List, Tuple

import torch
import triton


def bench(
    fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
    return_mode: str = "median",
) -> float:
    """Run fn many times and return time in milliseconds.

    Wraps triton.testing.do_bench. The default reduction is median, which is
    robust against the occasional slow run. Use "mean" if you specifically
    want the mean.
    """
    return triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, return_mode=return_mode
    )


def bench_stats(
    fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
) -> Dict[str, float]:
    """Return min, median, mean, max latency in ms for fn."""
    return {
        "min":    triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min"),
        "median": triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"),
        "mean":   triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean"),
        "max":    triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="max"),
    }


def bandwidth_gbs(bytes_total: int, ms: float) -> float:
    """Convert (bytes, milliseconds) to GB/s.

    1 GB/s = 1e9 bytes per second. Use this for memory-bound kernels.
    """
    return bytes_total / (ms * 1e-3) / 1e9


def tflops(flops_total: int, ms: float) -> float:
    """Convert (flops, milliseconds) to TFLOP/s.

    For matmul, flops_total = 2 * M * N * K (multiply-adds count as 2 FLOPs).
    Use this for compute-bound kernels.
    """
    return flops_total / (ms * 1e-3) / 1e12


def compare(
    fns: Dict[str, Callable[[], object]],
    warmup: int = 25,
    rep: int = 100,
) -> List[Tuple[str, float]]:
    """Time several functions and return (name, ms) sorted fastest first.

    Designed to feed straight into a Plotly bar chart. Example:
        results = compare({"triton": triton_fn,
                           "torch":  torch_fn,
                           "compiled": torch_compile_fn})
    """
    results = [(name, bench(fn, warmup=warmup, rep=rep)) for name, fn in fns.items()]
    results.sort(key=lambda kv: kv[1])
    return results


def _sync():
    """Manual cuda sync. Only needed for hand-rolled timing loops."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def manual_bench(
    fn: Callable[[], object],
    warmup: int = 5,
    iters: int = 50,
) -> float:
    """Hand-rolled cuda-synced timing loop. Use only as a reference / sanity
    check against do_bench; prefer do_bench for everything else.

    Returns mean ms.
    """
    for _ in range(warmup):
        fn()
    _sync()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters
