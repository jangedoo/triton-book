"""Row-wise (population) variance.

``out[i] = mean((x[i, :] - mean(x[i, :]))**2)``.

The naive identity ``Var(X) = E[X^2] - E[X]^2`` is faster (one pass) but
catastrophically unstable when ``E[X]^2`` is close to ``E[X^2]``: you lose
precision to cancellation. For LayerNorm we cannot accept that.

We use the two-pass approach instead:

1. Compute the mean across the row.
2. Loop again, accumulate the sum of squared deviations from the mean.
3. Divide by N.

A single kernel performs both passes — the row is reloaded from HBM for the
second pass. For the LayerNorm kernel in Chapter 6 we will revisit whether
to fuse this with a Welford accumulator (one pass, stable, slightly more
arithmetic per element). For now the two-pass version is the right shape:
clear, stable, and a direct stepping stone.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_variance_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * stride_xm

    # Pass 1: mean.
    sum_acc = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        sum_acc += tl.sum(x, axis=0)
    mean = sum_acc / N

    # Pass 2: sum of squared deviations.
    sq_acc = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        # other=mean is wrong here (you cannot pass a non-constant `other`),
        # so we use 0 and mask the diff explicitly.
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        sq_acc += tl.sum(diff * diff, axis=0)

    tl.store(out_ptr + pid_m, sq_acc / N)


def row_variance(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    row_variance_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out
