"""2-D transpose kernel.

Reads a ``[BM, BN]`` tile from ``x`` at ``(row, col)`` and writes it to ``out``
at ``(col, row)``. Demonstrates 2-D ``tl.arange`` broadcasting and asymmetric
strides on read vs write.

This is an *educational* kernel. Shared-memory tiling and swizzling for
bank-conflict avoidance is out of scope here — we will revisit those tricks
in Chapter 9 (matmul) and Chapter 14 (FlashAttention forward).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # block row in input
    pid_n = tl.program_id(axis=1)  # block col in input

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BN,)

    # 2-D mask, built by broadcasting two 1-D masks.
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    # Read tile from x at (rows, cols).
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    tile = tl.load(x_ptrs, mask=mask, other=0.0)

    # Write tile to out at (cols, rows). out is shape (N, M).
    out_ptrs = out_ptr + cols[None, :] * stride_om + rows[:, None] * stride_on
    tl.store(out_ptrs, tile, mask=mask)


def transpose(x: torch.Tensor, BLOCK_M: int = 32, BLOCK_N: int = 32) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out
