"""Worked solutions for Chapter 3 exercises."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise B1: strided 1-D copy
# ---------------------------------------------------------------------------
@triton.jit
def copy_strided_kernel(x_ptr, out_ptr, n_elements, sx, so, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets * sx, mask=mask)
    tl.store(out_ptr + offsets * so, x, mask=mask)


def copy_strided(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    copy_strided_kernel[grid](x, out, x.numel(), x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise B2: row scale
# ---------------------------------------------------------------------------
@triton.jit
def row_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    s = tl.load(scale_ptr + pid_m)  # scalar per program
    x_row = x_ptr + pid_m * stride_xm
    o_row = out_ptr + pid_m * stride_om
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0)
        tl.store(o_row + cols * stride_on, x * s, mask=mask)


def row_scale(x: torch.Tensor, scale: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    row_scale_kernel[grid](
        x, scale, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise B3: column-wise (per-row) add — bias of length M
# ---------------------------------------------------------------------------
@triton.jit
def col_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    b = tl.load(bias_ptr + pid_m)  # scalar
    x_row = x_ptr + pid_m * stride_xm
    o_row = out_ptr + pid_m * stride_om
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0)
        tl.store(o_row + cols * stride_on, x + b, mask=mask)


def col_add(x: torch.Tensor, bias: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    col_add_kernel[grid](
        x, bias, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise I1: strided vector_add
# ---------------------------------------------------------------------------
@triton.jit
def vector_add_strided_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    sx, sy, so,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets * sx, mask=mask)
    y = tl.load(y_ptr + offsets * sy, mask=mask)
    tl.store(out_ptr + offsets * so, x + y, mask=mask)


def vector_add_strided(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)  # contiguous output; tweak if you want strided output too.
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    vector_add_strided_kernel[grid](
        x, y, out, x.numel(),
        x.stride(0), y.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise I2: in-place row_add
# ---------------------------------------------------------------------------
@triton.jit
def row_add_inplace_kernel(
    x_ptr, bias_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * stride_xm
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        tl.store(x_row + cols * stride_xn, x + b, mask=mask)


def row_add_inplace(x: torch.Tensor, bias: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    grid = (M,)
    row_add_inplace_kernel[grid](
        x, bias,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return x


# ---------------------------------------------------------------------------
# Exercise A1: 2-D copy with arbitrary input strides
# ---------------------------------------------------------------------------
@triton.jit
def copy2d_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    o_ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on
    tile = tl.load(x_ptrs, mask=mask)
    tl.store(o_ptrs, tile, mask=mask)


def copy2d(x: torch.Tensor, BLOCK_M: int = 32, BLOCK_N: int = 32) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)  # contiguous
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    copy2d_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out
