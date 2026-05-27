"""
Educational attention backward.

NOT PRODUCTION. For S > 256 use torch.nn.functional.scaled_dot_product_attention,
which dispatches to a proper Flash backward.

The point of this file is to make the gradient math visible. Each gradient
(dV, dQ, dK) is a separate small kernel. The probability matrix P is
materialized in fp32 on the host side; production avoids this by recomputing
P tile-by-tile from the saved (O, log-sum-exp) pair.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _dv_kernel(
    p_ptr, do_ptr, dv_ptr,
    B, H, S, D,
    stride_pb, stride_ph, stride_pm, stride_pn,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dvb, stride_dvh, stride_dvm, stride_dvd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dV[b, h, n, d] = sum_m P[b, h, m, n] * dO[b, h, m, d]."""
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for m_start in range(0, S, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        p_ptrs = (
            p_ptr + b * stride_pb + h * stride_ph
            + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        )
        do_ptrs = (
            do_ptr + b * stride_dob + h * stride_doh
            + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        )
        m_mask = offs_m < S
        n_mask = offs_n < S
        p  = tl.load(p_ptrs,  mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=m_mask[:, None],                   other=0.0)
        acc += tl.dot(tl.trans(p), do.to(tl.float32))

    dv_ptrs = (
        dv_ptr + b * stride_dvb + h * stride_dvh
        + offs_n[:, None] * stride_dvm + offs_d[None, :] * stride_dvd
    )
    tl.store(dv_ptrs, acc, mask=(offs_n < S)[:, None])


@triton.jit
def _dq_kernel(
    ds_ptr, k_ptr, dq_ptr,
    B, H, S, D, scale,
    stride_sb, stride_sh, stride_sm, stride_sn,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_qb, stride_qh, stride_qm, stride_qd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dQ[b, h, m, d] = scale * sum_n dS[b, h, m, n] * K[b, h, n, d]."""
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n_start in range(0, S, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        ds_ptrs = (
            ds_ptr + b * stride_sb + h * stride_sh
            + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        )
        k_ptrs = (
            k_ptr + b * stride_kb + h * stride_kh
            + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        m_mask = offs_m < S
        n_mask = offs_n < S
        ds = tl.load(ds_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        k  = tl.load(k_ptrs,  mask=n_mask[:, None],                   other=0.0)
        acc += tl.dot(ds, k.to(tl.float32))

    acc = acc * scale
    dq_ptrs = (
        dq_ptr + b * stride_qb + h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    tl.store(dq_ptrs, acc, mask=(offs_m < S)[:, None])


@triton.jit
def _dk_kernel(
    ds_ptr, q_ptr, dk_ptr,
    B, H, S, D, scale,
    stride_sb, stride_sh, stride_sm, stride_sn,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dK[b, h, n, d] = scale * sum_m dS[b, h, m, n] * Q[b, h, m, d]."""
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for m_start in range(0, S, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        ds_ptrs = (
            ds_ptr + b * stride_sb + h * stride_sh
            + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        )
        q_ptrs = (
            q_ptr + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        m_mask = offs_m < S
        n_mask = offs_n < S
        ds = tl.load(ds_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        q  = tl.load(q_ptrs,  mask=m_mask[:, None],                   other=0.0)
        acc += tl.dot(tl.trans(ds), q.to(tl.float32))

    acc = acc * scale
    dk_ptrs = (
        dk_ptr + b * stride_kb + h * stride_kh
        + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    tl.store(dk_ptrs, acc, mask=(offs_n < S)[:, None])


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def attention_backward_educational(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Educational attention backward.

    Inputs: q, k, v, do all of shape [B, H, S, D], any float dtype.
    Returns: (dq, dk, dv) in fp32, same shape as the inputs.

    Limits: S <= 256 recommended; no causal mask; no dropout. Materializes
    [B, H, S, S] tensors for P and dP.
    """
    assert q.shape == k.shape == v.shape == do.shape, "q,k,v,do must match shapes"
    assert q.is_cuda, "cuda only"
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)

    q32  = q.to(torch.float32)
    k32  = k.to(torch.float32)
    v32  = v.to(torch.float32)
    do32 = do.to(torch.float32)

    # Forward pieces that backward needs:
    s  = (q32 @ k32.transpose(-1, -2)) * scale            # [B, H, S, S]
    p  = torch.softmax(s, dim=-1)                         # [B, H, S, S]
    dp = do32 @ v32.transpose(-1, -2)                     # [B, H, S, S]
    row = (dp * p).sum(dim=-1, keepdim=True)              # [B, H, S, 1]
    ds = p * (dp - row)                                   # [B, H, S, S]

    dq = torch.empty_like(q32)
    dk = torch.empty_like(k32)
    dv = torch.empty_like(v32)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(16, triton.next_power_of_2(D))

    grid_n = (B * H, triton.cdiv(S, BLOCK_N))
    grid_m = (B * H, triton.cdiv(S, BLOCK_M))

    _dv_kernel[grid_n](
        p, do32, dv,
        B, H, S, D,
        *p.stride(),  *do32.stride(),  *dv.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    _dq_kernel[grid_m](
        ds, k32, dq,
        B, H, S, D, scale,
        *ds.stride(), *k32.stride(),   *dq.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    _dk_kernel[grid_n](
        ds, q32, dk,
        B, H, S, D, scale,
        *ds.stride(), *q32.stride(),   *dk.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return dq, dk, dv
