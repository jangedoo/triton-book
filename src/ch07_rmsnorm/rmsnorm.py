"""Triton RMSNorm forward and backward.

RMSNorm:
    rms(x) = sqrt(mean(x**2) + eps)
    y = (x / rms(x)) * weight

Compared to LayerNorm: no mean subtraction, no bias. One reduction instead
of two. Used by LLaMA, Mistral, Gemma, Qwen, and most modern open-weights
LLMs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, y_ptr, w_ptr,
    rstd_ptr,           # *fp32: (M,), saved for backward
    x_row_stride, y_row_stride,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # fp32 accumulation. mean(x**2) overflows in fp16 for typical activations.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    sq = x * x
    sq = tl.where(mask, sq, 0.0)  # defensive; other=0.0 already zeros out-of-bounds
    mean_sq = tl.sum(sq, axis=0) / H
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    in_dtype = tl.load(x_row + cols, mask=mask, other=0.0).dtype
    tl.store(y_ptr + row * y_row_stride + cols, y.to(in_dtype), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm. Reduces over the last dim. Matches LLaMA semantics."""
    assert x.is_cuda
    orig_shape = x.shape
    H = orig_shape[-1]
    x2d = x.reshape(-1, H).contiguous()
    M = x2d.shape[0]
    y = torch.empty_like(x2d)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(H)
    assert BLOCK_SIZE <= 16384, f"hidden size {H} too large for single-tile kernel"
    num_warps = 4
    if BLOCK_SIZE >= 2048: num_warps = 8
    if BLOCK_SIZE >= 4096: num_warps = 16
    _rmsnorm_fwd_kernel[(M,)](
        x2d, y, weight,
        rstd,
        x2d.stride(0), y.stride(0),
        H, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)


# ---- Backward -----------------------------------------------------------
# d/dx [ (x / rms) * w ] for rms = sqrt(mean(x^2) + eps)
# Let r = rstd = 1/rms. Then:
#   y_i = x_i * r * w_i
#   dy_i/dx_j = r * w_i * (delta_ij + x_i * dr/dx_j)
# With dr/dx_j = -x_j / (H * rms^3) = -x_j * r^3 / H,
#   dL/dx_j = sum_i dL/dy_i * dy_i/dx_j
#           = r * w_j * dy_j - r^3 / H * x_j * sum_i (x_i * w_i * dy_i)
#           = r * (w_j * dy_j - x_j * r^2 * c)   where c = mean_i(x_i * w_i * dy_i)
# Simplify:  dx_j = r * w_j * dy_j - x_j * r * (1/H) * sum_i(x_i * w_i * dy_i * r^2)
# Or, since x_i * r = xhat_i:  dx_j = r * (w_j * dy_j) - xhat_j * (1/H) * sum_i(xhat_i * w_i * dy_i)


@triton.jit
def _rmsnorm_bwd_dx_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
    row_stride, H,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xhat = x * rstd
    wdy = w * dy
    c = tl.sum(xhat * wdy, axis=0) / H
    dx = rstd * wdy - xhat * c * rstd  # = rstd * (wdy - xhat^2 * <wdy*x>/H) — see derivation above
    # Rewriting in the cleanest equivalent form:
    # dx = rstd * (wdy - (1/H) * x * rstd^2 * sum_i(x_i * w_i * dy_i))
    # Note: xhat * c * rstd == x * rstd^2 * c == (1/H) * x * rstd^2 * sum_i(x_i*w_i*dy_i)
    tl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)


@triton.jit
def _rmsnorm_bwd_dw_kernel(
    dy_ptr, x_ptr, rstd_ptr, dw_ptr,
    row_stride, M, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < H
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r_start in range(0, M, BLOCK_M):
        rows = r_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        m2d = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + rows[:, None] * row_stride + cols[None, :], mask=m2d, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + rows[:, None] * row_stride + cols[None, :], mask=m2d, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        xhat = x * rstd[:, None]
        dw += tl.sum(dy * xhat, axis=0)
    tl.store(dw_ptr + cols, dw, mask=col_mask)


def rmsnorm_backward(dy, x, weight, rstd):
    """Returns (dx, dweight). `rstd` is saved from the forward."""
    M, H = x.shape
    dx = torch.empty_like(x, dtype=torch.float32)
    dw = torch.empty(H, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(H)
    _rmsnorm_bwd_dx_kernel[(M,)](dy, x, weight, rstd, dx, x.stride(0), H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(H, BLOCK_N),)
    _rmsnorm_bwd_dw_kernel[grid](dy, x, rstd, dw, x.stride(0), M, H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
    return dx, dw.to(weight.dtype)
