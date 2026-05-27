"""Benchmarks for Chapter 3: copy, row_add, transpose.

Reports median ms and effective GB/s for each kernel and compares against the
PyTorch reference.
"""

from __future__ import annotations

import torch
import triton

from src.ch03_memory import copy, row_add, transpose


DTYPE = torch.float32
BYTES = torch.finfo(DTYPE).bits // 8


def gbps(elems_traffic: int, ms: float) -> float:
    return elems_traffic * BYTES / (ms * 1e-3) / 1e9


def bench_copy() -> None:
    print("\n== copy ==")
    print(f"{'N':>10} {'triton_ms':>10} {'torch_ms':>10} {'triton_GB/s':>12} {'torch_GB/s':>12}")
    for n in [2**i for i in (14, 16, 18, 20, 22, 24)]:
        x = torch.randn(n, device="cuda", dtype=DTYPE)
        t_triton = triton.testing.do_bench(lambda: copy(x))
        t_torch = triton.testing.do_bench(lambda: x.clone())
        # Copy: 1 load + 1 store per element.
        traffic = 2 * n
        print(f"{n:>10} {t_triton:>10.4f} {t_torch:>10.4f} "
              f"{gbps(traffic, t_triton):>12.1f} {gbps(traffic, t_torch):>12.1f}")


def bench_row_add() -> None:
    print("\n== row_add ==")
    print(f"{'M':>6} {'N':>6} {'triton_ms':>10} {'torch_ms':>10} {'triton_GB/s':>12} {'torch_GB/s':>12}")
    for M, N in [(256, 1024), (512, 2048), (1024, 4096), (2048, 8192)]:
        x = torch.randn(M, N, device="cuda", dtype=DTYPE)
        b = torch.randn(N, device="cuda", dtype=DTYPE)
        t_triton = triton.testing.do_bench(lambda: row_add(x, b))
        t_torch = triton.testing.do_bench(lambda: x + b)
        # Each output element: 1 load from x, 1 load from bias (cached after first row), 1 store.
        traffic = 3 * M * N
        print(f"{M:>6} {N:>6} {t_triton:>10.4f} {t_torch:>10.4f} "
              f"{gbps(traffic, t_triton):>12.1f} {gbps(traffic, t_torch):>12.1f}")


def bench_transpose() -> None:
    print("\n== transpose ==")
    print(f"{'M':>6} {'N':>6} {'triton_ms':>10} {'torch_ms':>10} {'triton_GB/s':>12} {'torch_GB/s':>12}")
    for M, N in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]:
        x = torch.randn(M, N, device="cuda", dtype=DTYPE)
        t_triton = triton.testing.do_bench(lambda: transpose(x))
        t_torch = triton.testing.do_bench(lambda: x.t().contiguous())
        traffic = 2 * M * N
        print(f"{M:>6} {N:>6} {t_triton:>10.4f} {t_torch:>10.4f} "
              f"{gbps(traffic, t_triton):>12.1f} {gbps(traffic, t_torch):>12.1f}")


def main() -> None:
    assert torch.cuda.is_available(), "benchmarks require CUDA"
    bench_copy()
    bench_row_add()
    bench_transpose()


if __name__ == "__main__":
    main()
