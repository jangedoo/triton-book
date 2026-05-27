"""Chapter 25 — benchmarking utilities."""

from .benchmark_utils import (
    bench,
    bandwidth_gbs,
    tflops,
    compare,
    bench_stats,
)

__all__ = ["bench", "bandwidth_gbs", "tflops", "compare", "bench_stats"]
