"""Triton LayerNorm forward (with an optional backward kernel).

Semantics match `F.layer_norm(x, [hidden], weight, bias, eps)`:

    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    y    = (x - mean) / sqrt(var + eps) * weight + bias

We adopt PyTorch's eps placement: `1 / sqrt(var + eps)`, not `1 / (sqrt(var) + eps)`.

One program per row. Hidden dim must fit in one BLOCK_SIZE tile; on consumer
GPUs that comfortably covers hidden sizes up to 16384.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,          # *T: (M, H), row-contiguous
    y_ptr,          # *T: (M, H), output
    w_ptr,          # *T: (H,), gamma
    b_ptr,          # *T: (H,), beta
    mean_ptr,       # *fp32: (M,), saved mean (for backward)
    rstd_ptr,       # *fp32: (M,), saved rstd  (for backward)
    x_row_stride,   # int
    y_row_stride,   # int
    H,              # int (n_cols)
    eps,            # fp32
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Load to fp32 for all accumulations regardless of input dtype.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean. Out-of-bounds lanes loaded as 0.0 do not bias the sum,
    # but we divide by H (the true count), not BLOCK_SIZE.
    mean = tl.sum(x, axis=0) / H

    # Variance, unbiased=False (matches PyTorch default for layer_norm).
    xmu = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmu * xmu, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Save for backward.
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Affine transform.
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xmu * rstd * w + b

    # Cast back to the output dtype (= input dtype) on store.
    tl.store(y_row + cols, y.to(tl.load(x_row + cols, mask=mask, other=0.0).dtype), mask=mask)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Row-wise LayerNorm. Reduces over the last dim.

    Shapes:
        x      : (..., H)
        weight : (H,)
        bias   : (H,)
    Returns: same shape and dtype as `x`.
    """
    assert x.is_cuda
    orig_shape = x.shape
    H = orig_shape[-1]
    x2d = x.reshape(-1, H).contiguous()
    M = x2d.shape[0]
    y = torch.empty_like(x2d)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(H)
    assert BLOCK_SIZE <= 16384, f"hidden size {H} too large for single-tile kernel"
    num_warps = 4
    if BLOCK_SIZE >= 2048: num_warps = 8
    if BLOCK_SIZE >= 4096: num_warps = 16
    _layernorm_fwd_kernel[(M,)](
        x2d, y, weight, bias,
        mean, rstd,
        x2d.stride(0), y.stride(0),
        H, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)


# ---- Optional backward --------------------------------------------------
# Implemented for completeness. The backward of LayerNorm has three outputs
# (dx, dweight, dbias). dweight and dbias are reductions over the batch axis;
# we compute them with a separate atomic-add kernel for simplicity.


@triton.jit
def _layernorm_bwd_dx_kernel(
    dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
    row_stride, H,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x  = tl.load(x_ptr  + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    wdy = tl.where(mask, wdy, 0.0)

    c1 = tl.sum(xhat * wdy, axis=0) / H
    c2 = tl.sum(wdy,        axis=0) / H

    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(dx_ptr + row * row_stride + cols, dx.to(tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).dtype), mask=mask)


@triton.jit
def _layernorm_bwd_dwdb_kernel(
    dy_ptr, x_ptr, mean_ptr, rstd_ptr, dw_ptr, db_ptr,
    row_stride, M, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < H
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r_start in range(0, M, BLOCK_M):
        rows = r_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        m2d = row_mask[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + rows[:, None] * row_stride + cols[None, :], mask=m2d, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + rows[:, None] * row_stride + cols[None, :], mask=m2d, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        xhat = (x - mean[:, None]) * rstd[:, None]
        dw += tl.sum(dy * xhat, axis=0)
        db += tl.sum(dy, axis=0)
    tl.store(dw_ptr + cols, dw, mask=col_mask)
    tl.store(db_ptr + cols, db, mask=col_mask)


def layernorm_backward(dy, x, weight, mean, rstd):
    """Returns (dx, dweight, dbias) given saved (mean, rstd)."""
    assert x.is_cuda and dy.is_cuda
    M, H = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty(H, device=x.device, dtype=torch.float32)
    db = torch.empty(H, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(H)
    _layernorm_bwd_dx_kernel[(M,)](dy, x, weight, mean, rstd, dx, x.stride(0), H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    BLOCK_N = 64
    BLOCK_M = 64
    grid = (triton.cdiv(H, BLOCK_N),)
    _layernorm_bwd_dwdb_kernel[grid](dy, x, mean, rstd, dw, db, x.stride(0), M, H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
    return dx, dw.to(weight.dtype), db.to(weight.dtype)
