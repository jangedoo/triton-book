"""Benchmark Chapter 7 RMSNorm."""

from __future__ import annotations

import torch
import triton

from src.ch07_rmsnorm import rmsnorm


def torch_rmsnorm(x, w, eps=1e-6):
    x32 = x.to(torch.float32)
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * rstd * w.to(torch.float32)).to(x.dtype)


def bench_one(M: int, H: int, dtype=torch.float16):
    x = torch.randn(M, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype) * 0.1 + 1.0

    def run_torch():
        return torch_rmsnorm(x, w)

    compiled = torch.compile(torch_rmsnorm)
    compiled(x, w); torch.cuda.synchronize()

    def run_compiled():
        return compiled(x, w)

    def run_triton():
        return rmsnorm(x, w)

    ms_torch = triton.testing.do_bench(run_torch)
    ms_compiled = triton.testing.do_bench(run_compiled)
    ms_triton = triton.testing.do_bench(run_triton)

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
