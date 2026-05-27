"""SiLU / Swish activation: x * sigmoid(x)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * (1.0 / (1.0 + tl.exp(-x)))  # sigmoid(x) = 1/(1+exp(-x))
    in_dtype = tl.load(x_ptr + offs, mask=mask, other=0.0).dtype
    tl.store(y_ptr + offs, y.to(in_dtype), mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _silu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return y
