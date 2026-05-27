"""Fused SwiGLU activation. Lifted from Chapter 19.

Given gate and up projections `a` and `b` (the two halves of a SwiGLU MLP
projection), computes:

    y = silu(a) * b = a * sigmoid(a) * b

In practice the caller passes the two halves of a single Linear's output;
this kernel reads both, applies the activation, and writes the gated
result.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    a_ptr,
    b_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = a * tl.sigmoid(a)
    y = silu * b
    tl.store(y_ptr + offs, y, mask=mask)


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute `silu(a) * b` element-wise.

    `a` and `b` must broadcast to identical shapes (and we require they
    already do — this kernel does not broadcast). Result has the dtype of
    `a`.
    """
    if a.shape != b.shape:
        raise ValueError("swiglu: a and b must have the same shape")
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    y = torch.empty_like(a_flat)
    N = a_flat.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _swiglu_kernel[grid](a_flat, b_flat, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(a)
