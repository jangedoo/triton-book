"""Benchmark the Chapter 2 vector_add kernel.

Sweeps tensor size N and BLOCK_SIZE, reports median ms and effective GB/s,
and compares against PyTorch's ``a + b``.

Run with:

    python benchmarks/bench_ch02_vector_add.py
"""

from __future__ import annotations

import torch
import triton

from src.ch02_mental_model import vector_add


SIZES = [2**i for i in (12, 14, 16, 18, 20, 22, 24)]
BLOCK_SIZES = [64, 128, 256, 1024, 4096]
DTYPE = torch.float32
BYTES_PER_ELEM = torch.finfo(DTYPE).bits // 8


def gbps(n: int, ms: float) -> float:
    """Effective bandwidth: two loads + one store per element."""
    return 3 * n * BYTES_PER_ELEM / (ms * 1e-3) / 1e9


def bench_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    return triton.testing.do_bench(lambda: x + y)


def bench_triton(x: torch.Tensor, y: torch.Tensor, block_size: int) -> float:
    return triton.testing.do_bench(lambda: vector_add(x, y, BLOCK_SIZE=block_size))


def main() -> None:
    assert torch.cuda.is_available(), "benchmarks require CUDA"
    print(f"{'N':>10} {'BLOCK':>6} {'triton_ms':>10} {'torch_ms':>10} "
          f"{'triton_GB/s':>12} {'torch_GB/s':>12}")
    for n in SIZES:
        x = torch.randn(n, device="cuda", dtype=DTYPE)
        y = torch.randn(n, device="cuda", dtype=DTYPE)
        torch_ms = bench_torch(x, y)
        for bs in BLOCK_SIZES:
            triton_ms = bench_triton(x, y, bs)
            print(f"{n:>10} {bs:>6} {triton_ms:>10.4f} {torch_ms:>10.4f} "
                  f"{gbps(n, triton_ms):>12.1f} {gbps(n, torch_ms):>12.1f}")


if __name__ == "__main__":
    main()
