"""Benchmark Chapter 9 matmul against PyTorch eager and torch.compile.

Prints median ms and TFLOP/s for square M=N=K from 256 to 4096.
"""

from __future__ import annotations

import torch
import triton

from src.ch09_matmul import matmul_naive, matmul_grouped


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def run():
    assert torch.cuda.is_available(), "needs cuda"
    sizes = [256, 512, 1024, 2048, 4096]

    # torch.compile baseline.
    compiled = torch.compile(lambda x, y: torch.matmul(x, y), mode="max-autotune")

    print(f"{'M=N=K':>6} {'eager(ms)':>10} {'compile(ms)':>12} {'naive(ms)':>10} {'group(ms)':>10}  "
          f"{'eager TF':>8} {'compile TF':>10} {'naive TF':>8} {'group TF':>8}")
    for n in sizes:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)

        ms_eager = triton.testing.do_bench(lambda: torch.matmul(a, b))
        # warm up compile
        for _ in range(3):
            compiled(a, b)
        ms_comp = triton.testing.do_bench(lambda: compiled(a, b))
        ms_naive = triton.testing.do_bench(lambda: matmul_naive(a, b))
        ms_group = triton.testing.do_bench(lambda: matmul_grouped(a, b))

        print(f"{n:>6d} {ms_eager:>10.3f} {ms_comp:>12.3f} {ms_naive:>10.3f} {ms_group:>10.3f}  "
              f"{tflops(n,n,n,ms_eager):>8.1f} {tflops(n,n,n,ms_comp):>10.1f} "
              f"{tflops(n,n,n,ms_naive):>8.1f} {tflops(n,n,n,ms_group):>8.1f}")


if __name__ == "__main__":
    run()
