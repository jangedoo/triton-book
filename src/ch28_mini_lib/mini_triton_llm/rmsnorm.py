"""RMSNorm forward kernel. Lifted from Chapter 7.

Computes:

    y = x * rsqrt(mean(x ** 2, dim=-1, keepdim=True) + eps) * weight

One Triton program owns one row of `x`. Reductions and the rsqrt run in
fp32 regardless of the input dtype; the output is cast back to the input
dtype on store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_ptrs = x_ptr + row * stride_xm + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # mean of squares -> rsqrt
    sq = x * x
    mean = tl.sum(sq, axis=0) / N
    inv = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * inv * w

    y_ptrs = y_ptr + row * stride_ym + offs
    tl.store(y_ptrs, y, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm along the last dimension.

    Args:
        x: (..., N) tensor. fp16, bf16, or fp32. CUDA only.
        weight: (N,) tensor. Same dtype as `x` (or fp32, will be cast inside
            the kernel anyway).
        eps: small constant added to the mean-square for numerical stability.

    Returns:
        Tensor with the same shape and dtype as `x`.
    """
    if not x.is_cuda:
        raise ValueError("rmsnorm: x must be on CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"rmsnorm: unsupported dtype {x.dtype}")
    if weight.shape[-1] != x.shape[-1]:
        raise ValueError("rmsnorm: weight last dim must match x last dim")

    orig_shape = x.shape
    N = orig_shape[-1]
    x2d = x.reshape(-1, N).contiguous()
    M = x2d.shape[0]

    y = torch.empty_like(x2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_fwd_kernel[(M,)](
        x2d, weight, y,
        x2d.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,
    )
    return y.reshape(orig_shape)
