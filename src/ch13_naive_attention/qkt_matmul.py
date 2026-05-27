"""QK^T matmul for naive attention.

Treats (batch, head) as a single flattened leading dim. Each program
computes a [BLOCK_M, BLOCK_N] tile of the score matrix for one
(batch, head) slice.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def qkt_matmul_kernel(
    q_ptr, k_ptr, s_ptr,
    BH, S, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_sb, stride_sm, stride_sn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute one [BLOCK_M, BLOCK_N] tile of S = (Q @ K^T) * scale.

    Q and K live as flat [BH, S, D] tensors where BH = B * H.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + pid_b * stride_qb \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + pid_b * stride_kb \
        + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd

    mask_m = offs_m < S
    mask_n = offs_n < S
    mask_d = offs_d < D

    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # We want q @ k.T → [BLOCK_M, BLOCK_N]. k is [BLOCK_N, BLOCK_D]; transpose
    # via tl.trans so the inner product is over D.
    acc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale

    s_ptrs = s_ptr + pid_b * stride_sb \
        + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    tl.store(s_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def qkt_matmul(q: torch.Tensor, k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Launcher: return scores of shape [B, H, S, S] given Q, K of [B, H, S, D]."""
    assert q.shape == k.shape, "Q and K must share shape"
    assert q.is_cuda and k.is_cuda
    B, H, S, D = q.shape
    BH = B * H
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_flat = q.reshape(BH, S, D).contiguous()
    k_flat = k.reshape(BH, S, D).contiguous()
    s_flat = torch.empty((BH, S, S), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(S, BLOCK_M), triton.cdiv(S, BLOCK_N), BH)

    qkt_matmul_kernel[grid](
        q_flat, k_flat, s_flat,
        BH, S, D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return s_flat.reshape(B, H, S, S)
