"""Fused residual + RMSNorm kernel. Lifted from Chapter 18.

Computes:

    h = x + residual                       # also returned, for the next layer
    y = h * rsqrt(mean(h ** 2)) * weight

Doing the add inside the same program avoids round-tripping `h` through
HBM between layers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    h_ptr,
    y_ptr,
    stride_xm,
    stride_rm,
    stride_hm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * stride_rm + offs, mask=mask, other=0.0).to(tl.float32)
    h = x + r

    sq = h * h
    mean = tl.sum(sq, axis=0) / N
    inv = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = h * inv * w

    tl.store(h_ptr + row * stride_hm + offs, h, mask=mask)
    tl.store(y_ptr + row * stride_ym + offs, y, mask=mask)


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply `h = x + residual; y = rmsnorm(h, weight)` as one kernel.

    Returns `(h, y)`. `h` is needed as the residual stream for the next
    sublayer; `y` is the normalized value passed into attention or MLP.
    """
    if x.shape != residual.shape:
        raise ValueError("residual_rmsnorm: x and residual must match shape")
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    r2 = residual.reshape(-1, N).contiguous()
    M = x2.shape[0]

    h2 = torch.empty_like(x2)
    y2 = torch.empty_like(x2)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _residual_rmsnorm_kernel[(M,)](
        x2, r2, weight, h2, y2,
        x2.stride(0), r2.stride(0), h2.stride(0), y2.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,
    )
    return h2.reshape(x.shape), y2.reshape(x.shape)
