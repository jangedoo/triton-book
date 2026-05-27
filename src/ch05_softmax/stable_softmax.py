"""Numerically stable row-wise softmax.

Single-tile-per-row variant: the whole row must fit in one BLOCK_SIZE-wide
tile. For rows larger than ~16384 elements you want the online variant
in `online_softmax.py`.

Math:
    m = max(x)
    e = exp(x - m)
    y = e / sum(e)

`m` cancels in the division, so subtracting it does not change the answer
but does keep every `exp` argument <= 0, so no overflow.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stable_softmax_kernel(
    x_ptr,
    y_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start_ptr = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    # other=-inf so masked lanes do not poison the max.
    x = tl.load(row_start_ptr + col_offs, mask=mask, other=-float("inf"))
    x_f32 = x.to(tl.float32)

    m = tl.max(x_f32, axis=0)
    e = tl.exp(x_f32 - m)
    # exp(-inf) is 0 so masked lanes contribute 0 to the sum already, but
    # we re-mask defensively in case the input dtype upcast changed anything.
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(y_ptr + row_idx * row_stride + col_offs, y.to(x.dtype), mask=mask)


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax, numerically stable. Single tile per row."""
    assert x.is_cuda and x.ndim == 2, "expected a 2D CUDA tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Pick num_warps roughly proportional to the row size so wide rows get
    # more threads. These cutoffs are reasonable on sm_75+; autotune later.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    _stable_softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
