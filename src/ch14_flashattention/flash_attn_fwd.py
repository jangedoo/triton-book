"""Simplified FlashAttention forward kernel.

This is a teaching implementation. It mirrors the spirit of the
official Triton fused-attention tutorial but is trimmed to one
configuration (head_dim=64, causal switchable, fp16 in/out, fp32
accumulators). See `solutions/ch14_solutions.py` for variations.

Layout: Q, K, V are [B, H, S, D]. We flatten (B, H) to BH = B * H
and launch a grid of (num_q_blocks, BH). Each program owns one
BLOCK_M-row tile of Q for one (batch, head) and streams over the
K/V tiles, maintaining the online softmax state (m_i, l_i, acc).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BH, S,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program owns BLOCK_M rows of Q for one (batch, head)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Row indices owned by this program (global, within this (B, H) slice).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S

    # --- Load Q tile once into registers/SRAM ------------------------------
    q_ptrs = q_ptr + pid_b * stride_qb \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)  # [BLOCK_M, BLOCK_D]

    # --- Online softmax state ---------------------------------------------
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)  # running row max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running row denom
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)        # running numerator

    # --- Loop bounds: causal prunes future K blocks -----------------------
    if IS_CAUSAL:
        # Last K column this Q tile can possibly attend to is (pid_m+1)*BLOCK_M - 1.
        # Convert to a count of N tiles.
        n_end = ((pid_m + 1) * BLOCK_M + BLOCK_N - 1)
        n_end = tl.minimum(n_end, S)
    else:
        n_end = S

    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load K, V tiles.
        k_ptrs = k_ptr + pid_b * stride_kb \
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb \
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, BLOCK_D]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, BLOCK_D]

        # Scores: Q @ K^T * scale. fp32 acc throughout.
        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * scale  # [BLOCK_M, BLOCK_N]

        # Mask out-of-range key positions (S not multiple of BLOCK_N).
        s_ij = tl.where(mask_n[None, :], s_ij, float("-inf"))

        # Causal mask using global row/col indices.
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float("-inf"))

        # ---- Online softmax update --------------------------------------
        # New running max.
        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        # Correction factor for previous accumulator.
        alpha = tl.exp(m_i - m_ij)
        # Probabilities for this block.
        p_ij = tl.exp(s_ij - m_ij[:, None])
        # Update denominator.
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)
        # Update numerator: correct old acc, then add new p @ V.
        acc = alpha[:, None] * acc + tl.dot(p_ij.to(v.dtype), v).to(tl.float32)
        # Roll forward.
        m_i = m_ij

    # --- Normalize and store ----------------------------------------------
    # Guard against rows that saw no valid keys (e.g. row 0 with causal=False
    # and S < BLOCK_N edge cases is fine; the causal pid_m=0 case still has
    # at least one valid key). l_i can underflow to 0 only if every key was
    # masked, which our launcher prevents.
    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + pid_b * stride_ob \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


def flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Launcher: q, k, v are [B, H, S, D] fp16, returns [B, H, S, D] fp16.

    This teaching kernel assumes D is a power of two up to 128. The
    BLOCK_D constexpr is set to D directly.
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and q.dtype == torch.float16
    B, H, S, D = q.shape
    assert D in (32, 64, 128), "teaching kernel covers D in {32, 64, 128}"

    BH = B * H
    q_flat = q.reshape(BH, S, D).contiguous()
    k_flat = k.reshape(BH, S, D).contiguous()
    v_flat = v.reshape(BH, S, D).contiguous()
    o_flat = torch.empty_like(q_flat)

    scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(S, block_m), BH)

    flash_attn_fwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat,
        BH, S,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        scale,
        IS_CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )
    return o_flat.reshape(B, H, S, D)
