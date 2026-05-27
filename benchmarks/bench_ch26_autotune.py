"""Benchmarks for Chapter 26.

Demonstrates that the autotuner picks different configs for different
shapes, and that the autotuned matmul beats a fixed-config baseline at
the extreme ends of the shape range.
"""

import torch
import triton

from src.ch26_autotune import autotuned_softmax, autotuned_matmul
from src.ch26_autotune.autotuned_matmul import _matmul_kernel


def bench_softmax():
    print("=== autotuned softmax ===")
    print(f"{'N':>6} {'ms':>8} {'GB/s':>8}")
    for N in (128, 256, 512, 1024, 2048, 4096):
        x = torch.randn(1024, N, device="cuda", dtype=torch.float32)
        ms = triton.testing.do_bench(
            lambda: autotuned_softmax(x),
            warmup=25, rep=100, return_mode="median",
        )
        bytes_total = 1024 * N * 4 * 2  # read + write
        gbs = bytes_total / (ms * 1e-3) / 1e9
        print(f"{N:>6} {ms:>8.3f} {gbs:>8.1f}")


def bench_matmul():
    print("\n=== autotuned matmul ===")
    print(f"{'M':>5} {'N':>5} {'K':>5} {'ms':>8} {'TFLOP/s':>9}")
    for M, N, K in [(256, 256, 256), (1024, 1024, 1024),
                    (4096, 4096, 4096), (1, 4096, 4096), (64, 8192, 8192)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        ms = triton.testing.do_bench(
            lambda: autotuned_matmul(a, b),
            warmup=25, rep=100, return_mode="median",
        )
        flops = 2 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"{M:>5} {N:>5} {K:>5} {ms:>8.3f} {tflops:>9.2f}")

    # Show the picker's choices
    print("\n=== best configs found ===")
    print(_matmul_kernel.best_config)


if __name__ == "__main__":
    bench_softmax()
    bench_matmul()
