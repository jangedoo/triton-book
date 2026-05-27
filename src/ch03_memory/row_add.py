"""Row-wise bias add.

``out[i, j] = x[i, j] + bias[j]`` for an ``[M, N]`` matrix ``x`` and a
``(N,)`` vector ``bias``. One program per row. The block tiles along ``N``
and, when ``BLOCK_SIZE_N < N``, the program loops in ``N`` chunks.

This is the smallest non-trivial 2-D pattern. It introduces:

- A 2-D grid (here: ``(M,)`` because we put the inner loop inside the kernel,
  but the structure generalizes).
- Row strides versus element strides.
- An inner loop over column tiles, with a mask on the last tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_add_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE_N: tl.constexpr,
):
    """One program owns one row. Inner loop tiles along columns."""
    pid_m = tl.program_id(axis=0)

    # Pointer to the start of this row, for both input and output.
    x_row_ptr = x_ptr + pid_m * stride_xm
    o_row_ptr = out_ptr + pid_m * stride_om

    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        # bias is 1-D and contiguous in this design.
        x = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        tl.store(o_row_ptr + cols * stride_on, x + b, mask=mask)


def row_add(x: torch.Tensor, bias: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    assert x.is_cuda and bias.is_cuda
    assert x.dim() == 2, "row_add expects a 2-D input"
    assert bias.dim() == 1 and bias.shape[0] == x.shape[1]
    assert bias.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (M,)
    row_add_kernel[grid](
        x, bias, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out
