"""Stable row-wise softmax. Lifted from Chapter 5."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_ym,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=-float("inf"))
    x_f32 = x.to(tl.float32)
    m = tl.max(x_f32, axis=0)
    e = tl.exp(x_f32 - m)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(y_ptr + row * stride_ym + offs, y.to(x.dtype), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last dim."""
    if not x.is_cuda:
        raise ValueError("softmax: x must be on CUDA")
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    y = torch.empty_like(x2)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _softmax_kernel[(x2.shape[0],)](
        x2, y,
        x2.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,
    )
    return y.reshape(x.shape)
