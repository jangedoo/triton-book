"""Benchmarks for Chapter 4 reductions."""

from __future__ import annotations

import torch
import triton

from src.ch04_reductions import row_sum, row_max, row_mean, row_variance


DTYPE = torch.float32
BYTES = torch.finfo(DTYPE).bits // 8

SHAPES = [(512, 1024), (1024, 4096), (2048, 8192), (4096, 8192)]


def gbps(elems_traffic: int, ms: float) -> float:
    return elems_traffic * BYTES / (ms * 1e-3) / 1e9


def bench_one(name: str, kernel_fn, ref_fn, loads_per_elem: int) -> None:
    print(f"\n== {name} ==")
    print(f"{'M':>6} {'N':>6} {'triton_ms':>10} {'torch_ms':>10} {'triton_GB/s':>12} {'torch_GB/s':>12}")
    for M, N in SHAPES:
        x = torch.randn(M, N, device="cuda", dtype=DTYPE)
        t_triton = triton.testing.do_bench(lambda: kernel_fn(x))
        t_torch = triton.testing.do_bench(lambda: ref_fn(x))
        # Reductions: `loads_per_elem` loads per element + M scalar stores (negligible).
        traffic = loads_per_elem * M * N
        print(f"{M:>6} {N:>6} {t_triton:>10.4f} {t_torch:>10.4f} "
              f"{gbps(traffic, t_triton):>12.1f} {gbps(traffic, t_torch):>12.1f}")


def main() -> None:
    assert torch.cuda.is_available(), "benchmarks require CUDA"
    bench_one("row_sum", row_sum, lambda x: x.sum(dim=1), loads_per_elem=1)
    bench_one("row_max", row_max, lambda x: x.amax(dim=1), loads_per_elem=1)
    bench_one("row_mean", row_mean, lambda x: x.mean(dim=1), loads_per_elem=1)
    # row_variance reads the row twice (two-pass).
    bench_one("row_variance", row_variance, lambda x: x.var(dim=1, unbiased=False), loads_per_elem=2)


if __name__ == "__main__":
    main()
