"""Row-wise mean.

``out[i] = sum(x[i, :]) / N``. Same structure as row_sum with a final divide.

Mean is the easy half of "variance" — what makes variance hard is the
*combination* of mean and squared deviation in a numerically stable way.
See ``row_variance.py`` for that discussion.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_mean_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * stride_xm
    acc = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + pid_m, acc / N)


def row_mean(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    row_mean_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out
