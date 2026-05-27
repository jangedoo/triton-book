"""Benchmark batched matmul and fused linear+bias+GELU vs the unfused stack."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from src.ch10_batched_linear import batched_matmul, linear_bias_gelu


def run_batched():
    assert torch.cuda.is_available()
    print("== Batched matmul (B=8, square M=N=K) ==")
    print(f"{'N':>6} {'torch(ms)':>10} {'triton(ms)':>11}")
    for n in [128, 256, 512, 1024, 2048]:
        a = torch.randn((8, n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((8, n, n), device="cuda", dtype=torch.float16)
        ms_t = triton.testing.do_bench(lambda: torch.matmul(a, b))
        ms_tr = triton.testing.do_bench(lambda: batched_matmul(a, b))
        print(f"{n:>6d} {ms_t:>10.3f} {ms_tr:>11.3f}")


def run_linear_fusion():
    print("\n== Linear + bias + GELU (B=4, T=1024, in=out=N) ==")
    print(f"{'N':>6} {'unfused(ms)':>12} {'fused(ms)':>10} {'speedup':>8}")
    for n in [512, 1024, 2048, 4096]:
        x = torch.randn((4, 1024, n), device="cuda", dtype=torch.float16)
        w = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n,),  device="cuda", dtype=torch.float16)

        unfused = lambda: F.gelu(F.linear(x, w, b), approximate="tanh")
        fused = lambda: linear_bias_gelu(x, w, b)

        ms_u = triton.testing.do_bench(unfused)
        ms_f = triton.testing.do_bench(fused)
        print(f"{n:>6d} {ms_u:>12.3f} {ms_f:>10.3f} {ms_u/ms_f:>8.2f}x")


if __name__ == "__main__":
    run_batched()
    run_linear_fusion()
