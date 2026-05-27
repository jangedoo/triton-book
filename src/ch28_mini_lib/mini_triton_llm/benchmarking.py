"""Benchmark helpers. Adapted from Chapter 25.

Thin wrappers around `triton.testing.do_bench` plus a few unit-conversion
helpers and a `compare` function that runs two callables and prints a
side-by-side table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


def bench(fn: Callable[[], object], warmup: int = 25, rep: int = 100) -> float:
    """Median wall-time of `fn` in milliseconds.

    `fn` must be a zero-arg callable that runs the op once. We delegate to
    `triton.testing.do_bench`, which handles warmup and CUDA-graph capture.
    """
    import triton  # local import keeps the module importable on CPU-only
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def bandwidth_gbs(bytes_moved: int, ms: float) -> float:
    """Effective bandwidth in GB/s given bytes touched and time in ms."""
    if ms <= 0:
        return float("inf")
    return bytes_moved / (ms * 1e-3) / 1e9


def tflops(flops: int, ms: float) -> float:
    """Throughput in TFLOP/s given a FLOP count and time in ms."""
    if ms <= 0:
        return float("inf")
    return flops / (ms * 1e-3) / 1e12


@dataclass
class CompareResult:
    name: str
    triton_ms: float
    torch_ms: float

    @property
    def speedup(self) -> float:
        return self.torch_ms / self.triton_ms if self.triton_ms > 0 else float("inf")


def compare(
    name: str,
    triton_fn: Callable[[], object],
    torch_fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
) -> CompareResult:
    """Time `triton_fn` and `torch_fn` and return a CompareResult."""
    torch.cuda.synchronize()
    t = bench(triton_fn, warmup=warmup, rep=rep)
    p = bench(torch_fn, warmup=warmup, rep=rep)
    return CompareResult(name=name, triton_ms=t, torch_ms=p)


def print_table(rows: list[CompareResult]) -> None:
    """Print a CompareResult list as a fixed-width table."""
    print(f"{'kernel':<24} {'triton ms':>12} {'torch ms':>12} {'speedup':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r.name:<24} {r.triton_ms:>12.4f} {r.torch_ms:>12.4f} {r.speedup:>9.2f}x")
