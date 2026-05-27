"""Benchmark Chapter 6 LayerNorm.

Usage:
    python -m benchmarks.bench_ch06_layernorm
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from src.ch06_layernorm import layernorm


def bench_one(M: int, H: int, dtype=torch.float16):
    x = torch.randn(M, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype) * 0.1 + 1.0
    b = torch.randn(H, device="cuda", dtype=dtype) * 0.01

    def run_torch():
        return F.layer_norm(x, (H,), w, b, eps=1e-5)

    compiled = torch.compile(lambda t: F.layer_norm(t, (H,), w, b, eps=1e-5))
    compiled(x); torch.cuda.synchronize()

    def run_compiled():
        return compiled(x)

    def run_triton():
        return layernorm(x, w, b, eps=1e-5)

    ms_torch = triton.testing.do_bench(run_torch)
    ms_compiled = triton.testing.do_bench(run_compiled)
    ms_triton = triton.testing.do_bench(run_triton)

    # Memory: read x (M*H) + read w+b (2*H, tiny) + write y (M*H).
    bytes_moved = 2 * M * H * x.element_size()
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    print(f"M={M:>5} H={H:>5} | torch={ms_torch:.3f}ms ({gbps(ms_torch):.0f} GB/s) | "
          f"compile={ms_compiled:.3f}ms ({gbps(ms_compiled):.0f} GB/s) | "
          f"triton={ms_triton:.3f}ms ({gbps(ms_triton):.0f} GB/s)")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return
    for H in (768, 1024, 2048, 4096, 8192):
        bench_one(M=4096, H=H)


if __name__ == "__main__":
    main()
