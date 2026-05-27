"""Benchmark Chapter 5 softmax kernels against PyTorch.

Usage:
    python -m benchmarks.bench_ch05_softmax
"""

from __future__ import annotations

import torch
import triton

from src.ch05_softmax import stable_softmax, online_softmax


def bench_one(M: int, N: int, dtype=torch.float16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    def run_torch():
        return torch.softmax(x, dim=-1)

    compiled = torch.compile(lambda t: torch.softmax(t, dim=-1))
    # warm the compile cache
    compiled(x); torch.cuda.synchronize()

    def run_compiled():
        return compiled(x)

    def run_triton():
        return stable_softmax(x)

    def run_online():
        return online_softmax(x, block_size=1024)

    ms_torch = triton.testing.do_bench(run_torch)
    ms_compiled = triton.testing.do_bench(run_compiled)
    ms_triton = triton.testing.do_bench(run_triton) if N <= 16384 else float("nan")
    ms_online = triton.testing.do_bench(run_online)

    # Memory traffic: read x + write y, both M*N elements.
    bytes_moved = 2 * M * N * x.element_size()
    def gbps(ms): return bytes_moved / (ms * 1e-3) / 1e9 if ms == ms else float("nan")

    print(
        f"M={M:>5} N={N:>6} dtype={dtype} | "
        f"torch={ms_torch:.3f}ms ({gbps(ms_torch):.0f} GB/s) | "
        f"compile={ms_compiled:.3f}ms ({gbps(ms_compiled):.0f} GB/s) | "
        f"triton={ms_triton:.3f}ms ({gbps(ms_triton):.0f} GB/s) | "
        f"online={ms_online:.3f}ms ({gbps(ms_online):.0f} GB/s)"
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return
    for N in (256, 1024, 4096, 8192, 16384, 32768):
        bench_one(M=4096, N=N)


if __name__ == "__main__":
    main()
