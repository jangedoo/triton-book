"""PV matmul: probs @ V for naive attention.

Probs is [BH, S, S], V is [BH, S, D], output is [BH, S, D]. Each program
computes one [BLOCK_M, BLOCK_D] tile of the output for one (batch, head),
looping over the K dimension in tiles of BLOCK_N.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def pv_matmul_kernel(
    p_ptr, v_ptr, o_ptr,
    BH, S, D,
    stride_pb, stride_pm, stride_pn,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute one [BLOCK_M, BLOCK_D] tile of O = P @ V."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        p_ptrs = p_ptr + pid_b * stride_pb \
            + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        v_ptrs = v_ptr + pid_b * stride_vb \
            + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        p = tl.load(p_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc += tl.dot(p, v.to(p.dtype)).to(tl.float32)

    o_ptrs = o_ptr + pid_b * stride_ob \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_d[None, :])


def pv_matmul(probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Launcher: probs [B,H,S,S], v [B,H,S,D] -> out [B,H,S,D] (fp16)."""
    assert probs.is_cuda and v.is_cuda
    B, H, S, _ = probs.shape
    _, _, _, D = v.shape
    BH = B * H

    p_flat = probs.reshape(BH, S, S).contiguous()
    v_flat = v.reshape(BH, S, D).contiguous()
    o_flat = torch.empty((BH, S, D), device=v.device, dtype=torch.float16)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(S, BLOCK_M), BH)

    pv_matmul_kernel[grid](
        p_flat, v_flat, o_flat,
        BH, S, D,
        p_flat.stride(0), p_flat.stride(1), p_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o_flat.reshape(B, H, S, D)
