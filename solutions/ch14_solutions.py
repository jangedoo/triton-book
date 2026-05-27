"""Worked solutions for Chapter 14 exercises."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Exercise 1: Non-causal variant ------------------------------------------------
def exercise_non_causal():
    from src.ch14_flashattention import flash_attn_forward
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    got = flash_attn_forward(q, k, v, causal=False)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


# Exercise 3: Block-size sweep --------------------------------------------------
def exercise_block_sweep():
    from src.ch14_flashattention import flash_attn_forward
    B, H, S, D = 1, 4, 2048, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    for bm, bn in [(32, 32), (64, 64), (128, 64), (64, 128)]:
        ms = triton.testing.do_bench(
            lambda: flash_attn_forward(q, k, v, causal=True, block_m=bm, block_n=bn)
        )
        print(f"BLOCK_M={bm}, BLOCK_N={bn}: {ms:.3f} ms")


# Exercise 4: Arbitrary head_dim via padding ------------------------------------
@triton.jit
def flash_attn_padded_d_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BH, S, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    mask_d = offs_d < D

    q_ptrs = q_ptr + pid_b * stride_qb \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_end = ((pid_m + 1) * BLOCK_M + BLOCK_N - 1) if IS_CAUSAL else S
    n_end = tl.minimum(n_end, S) if IS_CAUSAL else S

    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        k_ptrs = k_ptr + pid_b * stride_kb \
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb \
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        s_ij = tl.where(mask_n[None, :], s_ij, float("-inf"))
        if IS_CAUSAL:
            s_ij = tl.where(offs_m[:, None] >= offs_n[None, :], s_ij, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p_ij.to(v.dtype), v).to(tl.float32)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_d[None, :])


def flash_attn_padded_d(q, k, v, causal=True):
    """Launcher that pads D up to next power of two inside the kernel."""
    B, H, S, D = q.shape
    BH = B * H
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M = 64
    BLOCK_N = 64

    q_flat = q.reshape(BH, S, D).contiguous()
    k_flat = k.reshape(BH, S, D).contiguous()
    v_flat = v.reshape(BH, S, D).contiguous()
    o_flat = torch.empty_like(q_flat)

    scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(S, BLOCK_M), BH)

    flash_attn_padded_d_kernel[grid](
        q_flat, k_flat, v_flat, o_flat,
        BH, S, D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o_flat.reshape(B, H, S, D)


# Exercise 5: Temperature -------------------------------------------------------
def flash_attn_with_temperature(q, k, v, temperature=1.0, causal=True):
    """Trivial wrapper: scale Q by 1/temperature before calling the kernel."""
    from src.ch14_flashattention import flash_attn_forward
    # Equivalent to dividing the score by temperature, which is what we want.
    return flash_attn_forward(q / float(temperature), k, v, causal=causal)


# Exercise 6: Variable-length sequences (partial) -------------------------------
@triton.jit
def flash_attn_varlen_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, seq_lens_ptr,
    BH, S, H,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Partial solution: applies per-batch seq_lens to mask both Q rows and K cols.

    Note: assumes the (B, H) flattening puts head as the inner dim so we
    can recover the batch index as pid_b // H.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    batch_idx = pid_b // H

    cur_len = tl.load(seq_lens_ptr + batch_idx)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = (offs_m < S) & (offs_m < cur_len)

    q_ptrs = q_ptr + pid_b * stride_qb \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_end = tl.minimum(cur_len, S)
    if IS_CAUSAL:
        n_end = tl.minimum(n_end, (pid_m + 1) * BLOCK_M + BLOCK_N - 1)

    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < S) & (offs_n < cur_len)
        k_ptrs = k_ptr + pid_b * stride_kb \
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + pid_b * stride_vb \
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        s_ij = tl.where(mask_n[None, :], s_ij, float("-inf"))
        if IS_CAUSAL:
            s_ij = tl.where(offs_m[:, None] >= offs_n[None, :], s_ij, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_i = alpha * l_i + tl.sum(p_ij, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p_ij.to(v.dtype), v).to(tl.float32)
        m_i = m_ij

    # Guard rows that saw no valid keys (padded positions): l_i == 0 there.
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None])


if __name__ == "__main__":
    exercise_non_causal()
    print("Non-causal variant matches SDPA.")
