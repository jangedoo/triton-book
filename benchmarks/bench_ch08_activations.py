"""Benchmark Chapter 8 activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from src.ch08_activations import gelu, silu, swiglu


def bench_one(N: int, dtype=torch.float16):
    x = torch.randn(N, device="cuda", dtype=dtype)
    g = torch.randn(N, device="cuda", dtype=dtype)
    u = torch.randn(N, device="cuda", dtype=dtype)

    def run_torch_gelu(): return F.gelu(x, approximate="tanh")
    compiled_gelu = torch.compile(lambda t: F.gelu(t, approximate="tanh")); compiled_gelu(x); torch.cuda.synchronize()
    def run_compiled_gelu(): return compiled_gelu(x)
    def run_triton_gelu(): return gelu(x, approximate="tanh")

    def run_torch_silu(): return F.silu(x)
    compiled_silu = torch.compile(F.silu); compiled_silu(x); torch.cuda.synchronize()
    def run_compiled_silu(): return compiled_silu(x)
    def run_triton_silu(): return silu(x)

    def run_torch_swi(): return F.silu(g) * u
    compiled_swi = torch.compile(lambda a, b: F.silu(a) * b); compiled_swi(g, u); torch.cuda.synchronize()
    def run_compiled_swi(): return compiled_swi(g, u)
    def run_triton_swi(): return swiglu(g, u)

    print(f"N={N}")
    for name, fn in [
        ("gelu torch ",    run_torch_gelu),
        ("gelu compile",   run_compiled_gelu),
        ("gelu triton",    run_triton_gelu),
        ("silu torch ",    run_torch_silu),
        ("silu compile",   run_compiled_silu),
        ("silu triton",    run_triton_silu),
        ("swi  torch ",    run_torch_swi),
        ("swi  compile",   run_compiled_swi),
        ("swi  triton",    run_triton_swi),
    ]:
        ms = triton.testing.do_bench(fn)
        bytes_moved = (2 if "swi" not in name else 3) * N * x.element_size()  # 1 read + 1 write, or 2 read + 1 write
        gbps = bytes_moved / (ms * 1e-3) / 1e9
        print(f"  {name}: {ms:.4f}ms ({gbps:.0f} GB/s)")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return
    for N in (1 << 16, 1 << 20, 1 << 24):
        bench_one(N)


if __name__ == "__main__":
    main()
