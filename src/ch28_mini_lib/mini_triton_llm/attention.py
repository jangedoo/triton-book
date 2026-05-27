"""Simplified FlashAttention forward. Lifted from Chapter 14.

Online-softmax streaming attention. One program owns one (batch, head,
query block) tuple. We iterate over key blocks, accumulate the running
max and the running denominator, and rescale the partial output on each
new block.

This is the *educational* version: no backward, no dropout, optional
causal masking, fp16/bf16 in, fp32 accumulate. Production code should
use the official `flash_attn` package.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    if IS_CAUSAL:
        n_end = tl.minimum((start_m + 1) * BLOCK_M, N)
    else:
        n_end = N

    for start_n in range(0, n_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        k_mask = n_idx[:, None] < N
        k_ptrs = k_base + (n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = v_base + (n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        # mask out-of-bounds and (optionally) future positions
        qk = tl.where(n_idx[None, :] < N, qk, -float("inf"))
        if IS_CAUSAL:
            causal = offs_m[:, None] >= n_idx[None, :]
            qk = tl.where(causal, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = o_base + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """FlashAttention forward.

    Args:
        q, k, v: (B, H, S, D). fp16 or bf16. D must be a power of two
            from 16 to 128.
        causal: whether to mask future positions.
        sm_scale: defaults to `1/sqrt(D)`.

    Returns:
        Output tensor with the same shape and dtype as `q`.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("flash_attention: q, k, v must share shape")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_attention: dtype must be fp16 or bf16")

    B, H, S, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _flash_attn_fwd[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return out
