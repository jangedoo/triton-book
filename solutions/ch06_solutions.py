"""Chapter 6 worked solutions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Exercise 1: fp32-only LayerNorm (intentionally fragile) -----------------

@triton.jit
def _ln_fp32_only_kernel(x_ptr, y_ptr, w_ptr, b_ptr, row_stride, H, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)  # NO cast
    mean = tl.sum(x, axis=0) / H
    xmu = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmu * xmu, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    y = xmu * rstd * w + b
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


def layernorm_fp32_only(x, w, b, eps=1e-5):
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and b.dtype == torch.float32
    M, H = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _ln_fp32_only_kernel[(M,)](x, y, w, b, x.stride(0), H, eps, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


# Exercise 2 & 3: affine flag --------------------------------------------

@triton.jit
def _ln_affine_kernel(x_ptr, y_ptr, w_ptr, b_ptr, row_stride, H, eps,
                       BLOCK_SIZE: tl.constexpr, AFFINE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    xmu = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmu * xmu, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xmu * rstd
    if AFFINE:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
    tl.store(y_ptr + row * row_stride + cols, y.to(tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).dtype), mask=mask)


def layernorm_optional_affine(x, weight=None, bias=None, eps=1e-5):
    affine = weight is not None
    if affine:
        assert bias is not None, "weight and bias must come as a pair"
    M, H = x.shape
    if not affine:
        # Pass dummy pointers; the kernel ignores them via AFFINE=False.
        weight = torch.empty(1, device=x.device, dtype=x.dtype)
        bias = torch.empty(1, device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _ln_affine_kernel[(M,)](x, y, weight, bias, x.stride(0), H, eps,
                              BLOCK_SIZE=BLOCK, AFFINE=affine, num_warps=4)
    return y


# Exercise 4: multi-dim normalized shape ---------------------------------

def layernorm_multi(x, normalized_shape, weight, bias, eps=1e-5):
    from src.ch06_layernorm import layernorm
    import math
    H = math.prod(normalized_shape)
    M = x.numel() // H
    x2d = x.reshape(M, H)
    w1d = weight.reshape(H)
    b1d = bias.reshape(H)
    y = layernorm(x2d, w1d, b1d, eps=eps)
    return y.reshape(x.shape)


# Exercise 5: backward via atomic dw/db (single kernel) -------------------

@triton.jit
def _ln_bwd_fused_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr,
                           dx_ptr, dw_ptr, db_ptr,
                           row_stride, H, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    xhat = (x - mean) * rstd
    wdy = tl.where(mask, w * dy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / H
    c2 = tl.sum(wdy, axis=0) / H
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(dx_ptr + row * row_stride + cols, dx, mask=mask)
    # Atomic accumulation of dw, db. Slow at high contention; OK for educational use.
    tl.atomic_add(dw_ptr + cols, dy * xhat, mask=mask)
    tl.atomic_add(db_ptr + cols, dy, mask=mask)


def layernorm_backward_atomic(dy, x, w, mean, rstd):
    M, H = x.shape
    dx = torch.empty_like(x, dtype=torch.float32)
    dw = torch.zeros(H, device=x.device, dtype=torch.float32)
    db = torch.zeros(H, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(H)
    _ln_bwd_fused_kernel[(M,)](dy, x, w, mean, rstd, dx, dw, db, x.stride(0), H, BLOCK_SIZE=BLOCK, num_warps=4)
    return dx, dw, db


# Exercise 6: fused dropout + LayerNorm ---------------------------------

@triton.jit
def _dropout_ln_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mask_ptr,
                         row_stride, H, p, seed, eps,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    in_bounds = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=in_bounds, other=0.0).to(tl.float32)
    # Per-element Philox RNG, deterministic from (seed, row, col).
    rand = tl.rand(seed, row * H + cols)
    keep = rand > p
    scale = 1.0 / (1.0 - p)
    x = tl.where(keep & in_bounds, x * scale, 0.0)
    # Save the mask (as 0/1) for backward.
    tl.store(mask_ptr + row * row_stride + cols, keep.to(tl.int8), mask=in_bounds)
    mean = tl.sum(x, axis=0) / H
    xmu = tl.where(in_bounds, x - mean, 0.0)
    var = tl.sum(xmu * xmu, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=in_bounds, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=in_bounds, other=0.0).to(tl.float32)
    y = xmu * rstd * w + b
    tl.store(y_ptr + row * row_stride + cols, y, mask=in_bounds)


def dropout_layernorm(x, weight, bias, p=0.1, seed=0, eps=1e-5):
    M, H = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mask = torch.empty(M, H, device=x.device, dtype=torch.int8)
    BLOCK = triton.next_power_of_2(H)
    _dropout_ln_kernel[(M,)](x, y, weight, bias, mask, x.stride(0), H, p, seed, eps, BLOCK_SIZE=BLOCK, num_warps=4)
    return y, mask


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print("Solutions module imports cleanly.")
