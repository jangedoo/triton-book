"""Single-token decode attention against a KV cache.

Q has shape [B, H, 1, D]. The kernel reads the first `current_len` rows of
the cache, runs an online softmax, and produces an output of shape
[B, H, 1, D].
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attn_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    B, H, CURRENT_LEN, D, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per (b, h). Streams over K/V tiles with online softmax."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh %  H

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_ptrs = q_ptr + b * stride_qb + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32) * scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(0, CURRENT_LEN, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < CURRENT_LEN

        k_ptrs = (
            k_cache_ptr + b * stride_kb + h * stride_kh
            + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_cache_ptr + b * stride_vb + h * stride_vh
            + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Scores: q is [BLOCK_D]; k_tile is [BLOCK_N, BLOCK_D]; result is [BLOCK_N].
        s = tl.sum(k_tile * q[None, :], axis=1)
        s = tl.where(n_mask, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    out_ptrs = out_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=d_mask)


def decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    current_len: int,
    block_n: int = 64,
) -> torch.Tensor:
    """Single-token decode attention.

    q:           [B, H, 1, D]
    k_cache:     [B, H, max_seq, D]
    v_cache:     [B, H, max_seq, D]
    current_len: number of valid tokens in the cache (must be >= 1)

    Returns: [B, H, 1, D] in the dtype of q.
    """
    assert q.is_cuda
    assert q.shape[2] == 1, "decode expects a single token query"
    B, H, _, D = q.shape
    assert k_cache.shape[0] == B and k_cache.shape[1] == H and k_cache.shape[3] == D
    assert v_cache.shape == k_cache.shape
    assert 1 <= current_len <= k_cache.shape[2]

    scale = 1.0 / math.sqrt(D)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    out = torch.empty_like(q)

    grid = (B * H,)
    _decode_attn_kernel[grid](
        q, k_cache, v_cache, out,
        B, H, current_len, D, scale,
        *q.stride(), *k_cache.stride(), *v_cache.stride(), *out.stride(),
        BLOCK_N=block_n, BLOCK_D=BLOCK_D,
    )
    return out
