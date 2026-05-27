"""Row-wise sum.

``out[i] = sum(x[i, :])`` for an ``[M, N]`` input. One program per row.

When ``N`` exceeds ``BLOCK_SIZE_N``, the kernel loops over column tiles and
accumulates a partial sum, then writes the final scalar.

Accumulation is done in fp32 regardless of input dtype. This matters for
fp16 / bf16: a naive in-place sum into a fp16 accumulator loses precision
very fast on long rows. The pattern ``acc = acc + tl.load(...).to(tl.float32)``
is the one you want to internalize for the rest of the book.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
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
        # other=0 is exactly right for sums — masked lanes contribute nothing.
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + pid_m, acc)


def row_sum(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    row_sum_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out
