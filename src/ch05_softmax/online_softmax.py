"""Online (multi-tile) softmax.

For rows wider than `BLOCK_SIZE`, we stream the row in tiles and maintain a
running max `m` and a running denominator `l`. When a new tile raises the
running max from `m_old` to `m_new`, we rescale the old contributions by
`alpha = exp(m_old - m_new)`. After one streaming pass we know the row's
true max and the true sum-of-exps; a second pass writes the normalized
output. This is exactly the trick FlashAttention uses, see Ch 14.

We implement it as a two-pass kernel because keeping the exponentials in
SRAM for the whole row is what the single-pass variant requires, and that
defeats the point of streaming.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(
    x_ptr,
    y_ptr,
    row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    row_start = x_ptr + row_idx * row_stride

    # ---- Pass 1: streaming reduction to find row max and sum-of-exps ----
    m = -float("inf")
    l = 0.0
    for tile_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        # alpha rescales the old `l` to the new pivot.
        alpha = tl.exp(m - m_new)
        # tile_sum is computed at the new pivot.
        tile_sum = tl.sum(tl.exp(x - m_new), axis=0)
        l = alpha * l + tile_sum
        m = m_new

    # ---- Pass 2: write normalized output ----
    out_row_start = y_ptr + row_idx * row_stride
    for tile_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - m) / l
        tl.store(out_row_start + col_offs, y.to(tl.load(row_start + col_offs, mask=mask, other=0.0).dtype), mask=mask)


def online_softmax(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Row-wise stable softmax via online accumulator. Handles arbitrarily wide rows."""
    assert x.is_cuda and x.ndim == 2, "expected a 2D CUDA tensor"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(block_size)
    y = torch.empty_like(x)
    _online_softmax_kernel[(n_rows,)](
        x, y,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y
