"""Bench the persistent matmul skeleton -- skips on sm < 80.

The skip is not because the kernel will refuse to compile; it is because
the win on Turing is too small (or negative) to be worth a benchmark
chart. Run this on Ampere or newer.
"""

from __future__ import annotations

import sys

import torch
import triton

from src.ch09_matmul import matmul_grouped
from src.ch11_persistent_matmul import persistent_matmul_skeleton


def run():
    if not torch.cuda.is_available():
        print("cuda required; skipping")
        return
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        print(f"compute capability {cap} < (8, 0); persistent matmul "
              "is not expected to help on this GPU. Skipping benchmark.")
        return

    print(f"{'M=N=K':>6} {'grouped(ms)':>12} {'persistent(ms)':>15} {'speedup':>8}")
    for n in [512, 1024, 2048, 4096, 8192]:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)

        ms_g = triton.testing.do_bench(lambda: matmul_grouped(a, b))
        ms_p = triton.testing.do_bench(lambda: persistent_matmul_skeleton(a, b))
        print(f"{n:>6d} {ms_g:>12.3f} {ms_p:>15.3f} {ms_g/ms_p:>7.2f}x")


if __name__ == "__main__":
    run()
