"""Benchmark the fused residual + LayerNorm + dropout kernel against the
unfused PyTorch sequence.
"""

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "ch28_mini_lib"))

from ch30_playbook import fused_residual_ln_dropout  # noqa: E402
from mini_triton_llm.benchmarking import bench  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        print("cuda only; skipping")
        return

    rows = []
    for N in [1024, 2048, 4096, 8192]:
        M = 4096
        x = torch.randn(M, N, dtype=torch.float16, device="cuda")
        r = torch.randn(M, N, dtype=torch.float16, device="cuda")
        w = torch.randn(N, dtype=torch.float16, device="cuda")
        b = torch.randn(N, dtype=torch.float16, device="cuda")
        drop = torch.nn.Dropout(p=0.1)

        def torch_fn() -> torch.Tensor:
            h = x + r
            return drop(F.layer_norm(h.float(), (N,), w.float(), b.float()).to(torch.float16))

        def triton_fn() -> torch.Tensor:
            _, y = fused_residual_ln_dropout(x, r, w, b, p=0.1, seed=0)
            return y

        t_torch = bench(torch_fn)
        t_tri = bench(triton_fn)
        rows.append((N, t_torch, t_tri))

    print(f"{'N':<8} {'torch ms':>12} {'triton ms':>12} {'speedup':>10}")
    print("-" * 46)
    for N, tt, tr in rows:
        print(f"{N:<8} {tt:>12.4f} {tr:>12.4f} {tt / tr:>9.2f}x")


if __name__ == "__main__":
    main()
