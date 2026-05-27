"""Worked solutions for Chapter 2 exercises.

Each solution is runnable. The kernels copy the structure of
``src/ch02_mental_model/vector_add.py`` and change only the body of the
computation, except for A1 which introduces strides as a teaser for Chapter 3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise B1: vector subtraction
# ---------------------------------------------------------------------------
@triton.jit
def vector_sub_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x - y, mask=mask)


def vector_sub(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    vector_sub_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise B2: scalar multiply
# ---------------------------------------------------------------------------
@triton.jit
def scalar_mul_kernel(x_ptr, alpha, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * alpha, mask=mask)


def scalar_mul(x: torch.Tensor, alpha: float, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    scalar_mul_kernel[grid](x, alpha, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise B3: fused multiply-add
# ---------------------------------------------------------------------------
@triton.jit
def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y + z, mask=mask)


def fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    fma_kernel[grid](x, y, z, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise I1: clamp
# ---------------------------------------------------------------------------
@triton.jit
def clamp_kernel(x_ptr, out_ptr, lo, hi, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.maximum(x, lo)
    x = tl.minimum(x, hi)
    tl.store(out_ptr + offsets, x, mask=mask)


def clamp(x: torch.Tensor, lo: float, hi: float, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    clamp_kernel[grid](x, out, lo, hi, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise I2: where / conditional copy
# ---------------------------------------------------------------------------
@triton.jit
def where_kernel(cond_ptr, x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cond = tl.load(cond_ptr + offsets, mask=mask).to(tl.int1)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(cond, x, y), mask=mask)


def where(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    where_kernel[grid](cond, x, y, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Exercise A1: strided vector add (1-D stride only — teaser for Ch 3)
# ---------------------------------------------------------------------------
@triton.jit
def vector_add_strided_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    stride_x, stride_y, stride_o,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets * stride_x, mask=mask)
    y = tl.load(y_ptr + offsets * stride_y, mask=mask)
    tl.store(out_ptr + offsets * stride_o, x + y, mask=mask)


def vector_add_strided(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    assert x.dim() == 1 and y.dim() == 1
    out = torch.empty_like(x)  # out is contiguous (stride 1)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    vector_add_strided_kernel[grid](
        x, y, out,
        x.numel(),
        x.stride(0), y.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
