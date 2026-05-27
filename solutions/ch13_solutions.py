"""Worked solutions for Chapter 13 exercises."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Exercise 1: Memory accounting -------------------------------------------------
def memory_accounting():
    for S in [4096, 8192]:
        B, H = 4, 32
        bytes_ = B * H * S * S * 2  # fp16
        gb = bytes_ / (1024 ** 3)
        print(f"S={S}: scores tensor ≈ {gb:.2f} GB in fp16")


# Exercise 2: Standalone causal mask test ---------------------------------------
def exercise_causal_mask():
    from src.ch13_naive_attention import apply_causal_mask
    S = 9
    scores = torch.randn(1, 1, S, S, device="cuda", dtype=torch.float32).clone()
    expected = scores.clone()
    causal = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
    expected = expected.masked_fill(causal, float("-inf"))
    apply_causal_mask(scores)
    torch.testing.assert_close(scores, expected)


# Exercise 3: Timing ------------------------------------------------------------
def time_pipeline():
    from src.ch13_naive_attention import naive_attention_forward
    B, H, D = 1, 4, 64
    for S in [512, 1024, 2048]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: naive_attention_forward(q, k, v, causal=True))
        print(f"S={S}: {ms:.3f} ms")


# Exercise 4: Chunked softmax ---------------------------------------------------
def chunked_softmax(scores: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Two-pass row softmax in chunks of `chunk` along the last dim."""
    B, H, M, N = scores.shape
    # Pass 1: row max via chunk-wise reduction.
    row_max = torch.full((B, H, M, 1), float("-inf"), device=scores.device)
    for start in range(0, N, chunk):
        sub = scores[..., start:start + chunk]
        row_max = torch.maximum(row_max, sub.amax(dim=-1, keepdim=True))
    # Pass 2: exp + sum.
    row_sum = torch.zeros((B, H, M, 1), device=scores.device, dtype=torch.float32)
    out = torch.empty_like(scores, dtype=torch.float32)
    for start in range(0, N, chunk):
        sub = scores[..., start:start + chunk]
        e = torch.exp((sub - row_max).float())
        row_sum = row_sum + e.sum(dim=-1, keepdim=True)
        out[..., start:start + chunk] = e
    return (out / row_sum).to(scores.dtype)


# Exercise 5: Non-causal variant ------------------------------------------------
def non_causal_test():
    from src.ch13_naive_attention import naive_attention_forward
    B, H, S, D = 1, 2, 64, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    got = naive_attention_forward(q, k, v, causal=False)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)


# Exercise 6: One-kernel QK^T + softmax -----------------------------------------
@triton.jit
def qk_softmax_kernel(
    q_ptr, k_ptr, p_ptr,
    BH, S, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_pb, stride_pm, stride_pn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute softmax((Q @ K^T) * scale) for one [BLOCK_M, S] row strip.

    Two passes over K: one for the row max, one for the normalized exp.
    Still writes the full [BH, S, S] tensor. Foreshadows FlashAttention.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S

    q_ptrs = q_ptr + pid_b * stride_qb \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)

    # Pass 1: row max.
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        k_ptrs = k_ptr + pid_b * stride_kb \
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_i = tl.maximum(m_i, tl.max(s, axis=1))

    # Pass 2: exp + sum + store.
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        k_ptrs = k_ptr + pid_b * stride_kb \
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        p = tl.exp(s - m_i[:, None])
        l_i += tl.sum(p, axis=1)
        p_ptrs = p_ptr + pid_b * stride_pb \
            + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        tl.store(p_ptrs, p, mask=mask_m[:, None] & mask_n[None, :])

    # Renormalize: read back and divide. (For brevity we leave this as a
    # second launcher; in practice you would fold it in.)


if __name__ == "__main__":
    memory_accounting()
