"""Benchmark the Chapter 13 naive attention pipeline vs PyTorch SDPA.

Expect PyTorch's SDPA to win by a wide margin; it dispatches to a Flash
kernel by default. The point of this script is to quantify just how
much HBM traffic on the [B, H, S, S] tensor costs you.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from src.ch13_naive_attention import naive_attention_forward


def bench(B: int = 2, H: int = 8, D: int = 64, causal: bool = True) -> None:
    print(f"{'S':>6} {'naive ms':>12} {'sdpa ms':>12} {'naive_mem_gb':>14}")
    for S in [256, 512, 1024, 2048]:
        try:
            q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

            naive_ms = triton.testing.do_bench(
                lambda: naive_attention_forward(q, k, v, causal=causal)
            )
            sdpa_ms = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            )
            # Score tensor in fp32 bytes, in GB.
            naive_mem_gb = B * H * S * S * 4 / (1024 ** 3)
            print(f"{S:>6} {naive_ms:>12.3f} {sdpa_ms:>12.3f} {naive_mem_gb:>14.3f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>6} OOM (naive materializes [B,H,S,S]; try smaller B or H)")


if __name__ == "__main__":
    bench()
