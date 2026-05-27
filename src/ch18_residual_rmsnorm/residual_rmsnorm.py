"""Fused residual-add + RMSNorm kernel.

Implements y, x_new = rmsnorm(x + residual), with optional gamma weight and
optional skip-residual output. One program per row. fp32 accumulation in the
variance reduction. See Chapter 18.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def residual_rmsnorm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-6,
):
    """Plain PyTorch oracle. Returns (y, x_new)."""
    x_new = x + residual
    x32 = x_new.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    y = x32 * inv_rms
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(x.dtype), x_new


@triton.jit
def _residual_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, y_ptr, x_new_ptr,
    stride_x_row, stride_r_row, stride_y_row, stride_xn_row,
    N, eps,
    HAS_WEIGHT: tl.constexpr,
    STORE_X_NEW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_r_row + cols, mask=mask, other=0.0).to(tl.float32)
    t = x + r

    if STORE_X_NEW:
        tl.store(x_new_ptr + row * stride_xn_row + cols, t, mask=mask)

    var = tl.sum(t * t, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(var + eps)
    y = t * inv_rms

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w

    tl.store(y_ptr + row * stride_y_row + cols, y, mask=mask)


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    return_residual: bool = True,
):
    """Fused y = rmsnorm(x + residual). If return_residual, also returns x + residual."""
    assert x.shape == residual.shape, "x and residual must share shape"
    assert x.is_cuda and residual.is_cuda, "tensors must be on cuda"
    assert x.dtype == residual.dtype, "x and residual must share dtype"

    *batch, N = x.shape
    M = 1
    for d in batch:
        M *= d
    x2 = x.reshape(M, N)
    r2 = residual.reshape(M, N)

    y = torch.empty_like(x2)
    x_new = torch.empty_like(x2) if return_residual else None

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _residual_rmsnorm_kernel[grid](
        x2, r2, weight, y, x_new if return_residual else x2,
        x2.stride(0), r2.stride(0), y.stride(0),
        x_new.stride(0) if return_residual else 0,
        N, eps,
        HAS_WEIGHT=weight is not None,
        STORE_X_NEW=return_residual,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    y = y.view_as(x)
    if return_residual:
        return y, x_new.view_as(x)
    return y
