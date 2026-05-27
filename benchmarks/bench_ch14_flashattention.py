"""Benchmark FlashAttention forward vs Ch13 naive vs PyTorch SDPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from src.ch13_naive_attention import naive_attention_forward
from src.ch14_flashattention import flash_attn_forward


def bench(B: int = 2, H: int = 8, D: int = 64, causal: bool = True) -> None:
    print(f"{'S':>6} {'naive ms':>12} {'flash ms':>12} {'sdpa ms':>12}")
    for S in [256, 512, 1024, 2048, 4096]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        flash_ms = triton.testing.do_bench(
            lambda: flash_attn_forward(q, k, v, causal=causal)
        )
        sdpa_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        )
        try:
            naive_ms = triton.testing.do_bench(
                lambda: naive_attention_forward(q, k, v, causal=causal)
            )
            naive_str = f"{naive_ms:>12.3f}"
        except torch.cuda.OutOfMemoryError:
            naive_str = f"{'OOM':>12}"

        print(f"{S:>6} {naive_str} {flash_ms:>12.3f} {sdpa_ms:>12.3f}")


if __name__ == "__main__":
    bench()
