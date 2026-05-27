"""Row-wise max.

``out[i] = max(x[i, :])``. One program per row, inner loop over column tiles.

The interesting twist versus ``row_sum`` is the ``other=`` value on the
masked load. For a sum, you want ``other=0`` (the additive identity). For a
max, you want ``other=-inf`` so masked lanes never win the comparison. Use
``-float("inf")`` at the Python level; Triton casts it to the appropriate
dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * stride_xm
    # Initialize the running max to the additive identity for max: -inf.
    NEG_INF = float("-inf")
    acc = tl.full((), NEG_INF, dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * stride_xn, mask=mask, other=NEG_INF).to(tl.float32)
        acc = tl.maximum(acc, tl.max(x, axis=0))
    tl.store(out_ptr + pid_m, acc)


def row_max(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    row_max_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out
