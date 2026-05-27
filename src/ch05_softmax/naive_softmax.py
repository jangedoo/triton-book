"""Naive row-wise softmax with no max subtraction.

Kept around for pedagogy. It will overflow for fp16 inputs whose row maxima
exceed ~11.09 (since exp(11.09) ~= 65504, the max finite fp16). Do not use
this in production code.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_softmax_kernel(
    x_ptr,          # *fp16/bf16/fp32: pointer to input
    y_ptr,          # *same dtype: pointer to output
    row_stride,     # int: elements between two consecutive rows
    n_cols,         # int: actual number of columns per row
    BLOCK_SIZE: tl.constexpr,  # int: power-of-two tile width >= n_cols
):
    # One program per row.
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # Load with 0.0 default. That is fine here because we are NOT doing a
    # max reduction; the only reductions are exp (per-element) and sum.
    x = tl.load(row_start_ptr + col_offs, mask=mask, other=0.0)

    # Naive: exp without shifting. Accumulate in fp32 to delay the inevitable
    # overflow for inputs that are even close to the fp16 ceiling.
    e = tl.exp(x.to(tl.float32))
    # Mask off out-of-bounds lanes so they contribute zero to the sum.
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(y_ptr + row_idx * row_stride + col_offs, y.to(x.dtype), mask=mask)


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax. Single tile per row, no max subtraction.

    Caller's responsibility: ensure n_cols <= 16384 or so (we need the whole
    row to fit in one BLOCK_SIZE-wide tile).
    """
    assert x.is_cuda and x.ndim == 2, "expected a 2D CUDA tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _naive_softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y
