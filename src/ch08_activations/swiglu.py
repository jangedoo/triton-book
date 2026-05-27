"""SwiGLU and GEGLU gated activations.

SwiGLU: silu(x_gate) * x_up
GEGLU:  gelu(x_gate) * x_up

Both take two same-shape tensors and produce one output of the same shape.
Note that in real LLM MLPs the two halves come from a single linear projection
sliced in half; we keep them separate here so the kernel signature is clear.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


@triton.jit
def _swiglu_kernel(
    g_ptr, u_ptr, y_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
    y = silu_g * u
    in_dtype = tl.load(g_ptr + offs, mask=mask, other=0.0).dtype
    tl.store(y_ptr + offs, y.to(in_dtype), mask=mask)


def swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    assert x_gate.shape == x_up.shape and x_gate.dtype == x_up.dtype
    assert x_gate.is_cuda and x_up.is_cuda
    y = torch.empty_like(x_gate)
    N = x_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _swiglu_kernel[grid](x_gate, x_up, y, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return y


@triton.jit
def _geglu_kernel(
    g_ptr, u_ptr, y_ptr, N,
    BLOCK_SIZE: tl.constexpr,
    INV_SQRT2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gelu_g = 0.5 * g * (1.0 + tl.erf(g * INV_SQRT2))
    y = gelu_g * u
    in_dtype = tl.load(g_ptr + offs, mask=mask, other=0.0).dtype
    tl.store(y_ptr + offs, y.to(in_dtype), mask=mask)


def geglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    assert x_gate.shape == x_up.shape and x_gate.dtype == x_up.dtype
    assert x_gate.is_cuda and x_up.is_cuda
    y = torch.empty_like(x_gate)
    N = x_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _geglu_kernel[grid](x_gate, x_up, y, N, BLOCK_SIZE=BLOCK_SIZE, INV_SQRT2=_INV_SQRT2, num_warps=4)
    return y
