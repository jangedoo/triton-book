"""Solutions for Chapter 18 exercises."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Exercise 1 (Beginner): Vanilla return — only y, no x_new.
@triton.jit
def _vanilla_kernel(
    x_ptr, r_ptr, w_ptr, y_ptr,
    stride_x, stride_r, stride_y,
    N, eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    t = x + r
    var = tl.sum(t * t, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    y = t * inv
    if HAS_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def residual_rmsnorm_yonly(x, r, w=None, eps=1e-6):
    M, N = x.reshape(-1, x.shape[-1]).shape
    x2 = x.reshape(M, N)
    r2 = r.reshape(M, N)
    y = torch.empty_like(x2)
    BS = triton.next_power_of_2(N)
    _vanilla_kernel[(M,)](
        x2, r2, w, y,
        x2.stride(0), r2.stride(0), y.stride(0),
        N, eps,
        HAS_WEIGHT=w is not None,
        BLOCK_SIZE=BS,
    )
    return y.view_as(x)


# Exercise 2 (Beginner): both outputs — reuse the chapter kernel directly.
from ch18_residual_rmsnorm import residual_rmsnorm as _chapter_kernel


def both_outputs(x, r, w):
    return _chapter_kernel(x, r, w, return_residual=True)


# Exercise 3 (Beginner): no gamma — pass weight=None.
def no_gamma(x, r):
    return _chapter_kernel(x, r, None)


# Intermediate 1: drop_residual flag.
@triton.jit
def _drop_residual_kernel(
    x_ptr, r_ptr, w_ptr, y_ptr,
    stride_x, stride_r, stride_y,
    N, eps,
    HAS_WEIGHT: tl.constexpr,
    DROP_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if DROP_RESIDUAL:
        t = x
    else:
        r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        t = x + r
    var = tl.sum(t * t, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    y = t * inv
    if HAS_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def maybe_residual_rmsnorm(x, r, w, eps=1e-6, drop_residual=False):
    M, N = x.reshape(-1, x.shape[-1]).shape
    x2 = x.reshape(M, N)
    r2 = r.reshape(M, N) if r is not None else x2  # unused when DROP_RESIDUAL
    y = torch.empty_like(x2)
    BS = triton.next_power_of_2(N)
    _drop_residual_kernel[(M,)](
        x2, r2, w, y,
        x2.stride(0), r2.stride(0), y.stride(0),
        N, eps,
        HAS_WEIGHT=w is not None,
        DROP_RESIDUAL=drop_residual,
        BLOCK_SIZE=BS,
    )
    return y.view_as(x)


# Intermediate 2: fused dropout between add and norm.
@triton.jit
def _dropout_kernel(
    x_ptr, r_ptr, w_ptr, y_ptr,
    stride_x, stride_r, stride_y,
    N, eps, p, seed,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    t = x + r
    rand = tl.rand(seed, row * BLOCK_SIZE + cols)
    keep = rand > p
    scale = 1.0 / (1.0 - p)
    t = tl.where(keep, t * scale, 0.0)
    var = tl.sum(t * t, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    y = t * inv
    if HAS_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def dropout_residual_rmsnorm(x, r, w, p, seed=0, eps=1e-6):
    M, N = x.reshape(-1, x.shape[-1]).shape
    x2 = x.reshape(M, N)
    r2 = r.reshape(M, N)
    y = torch.empty_like(x2)
    BS = triton.next_power_of_2(N)
    _dropout_kernel[(M,)](
        x2, r2, w, y,
        x2.stride(0), r2.stride(0), y.stride(0),
        N, eps, p, seed,
        HAS_WEIGHT=w is not None,
        BLOCK_SIZE=BS,
    )
    return y.view_as(x)


# Advanced: backward kernel sketch.
# Given y = w * (t * inv_rms) with inv_rms = rsqrt(mean(t^2) + eps), t = x + r:
#   dt = inv_rms * w * dy
#        - (1/N) * t * inv_rms^3 * sum(w * dy * t)
#   dx = dr = dt
#   dw = sum_rows(dy * t * inv_rms)
@triton.jit
def _backward_kernel(
    x_ptr, r_ptr, w_ptr, dy_ptr,
    dx_ptr, dr_ptr, dw_partial_ptr,
    stride_x, stride_r, stride_dy, stride_dx, stride_dr, stride_dwp,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    t = x + r
    mean_sq = tl.sum(t * t, axis=0) / N
    inv = 1.0 / tl.sqrt(mean_sq + eps)
    wdy = w * dy
    # dt = inv * wdy - t * inv^3 * mean(t * wdy)
    mean_twdy = tl.sum(t * wdy, axis=0) / N
    dt = inv * wdy - t * inv * inv * inv * mean_twdy

    tl.store(dx_ptr + row * stride_dx + cols, dt, mask=mask)
    tl.store(dr_ptr + row * stride_dr + cols, dt, mask=mask)
    # partial dw per row; sum on the host.
    tl.store(dw_partial_ptr + row * stride_dwp + cols, dy * t * inv, mask=mask)
