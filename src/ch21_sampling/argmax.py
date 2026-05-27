"""Per-row argmax (greedy sampling). See Chapter 21."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_kernel(
    x_ptr, out_ptr,
    N, V, stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=-float("inf"))
    idx = tl.argmax(x, axis=0)
    tl.store(out_ptr + row, idx.to(tl.int64))


def argmax_sample(logits):
    """idx[row] = argmax over vocab. Returns int64."""
    assert logits.is_cuda
    N, V = logits.shape
    out = torch.empty(N, device=logits.device, dtype=torch.int64)
    BS = triton.next_power_of_2(V)
    _argmax_kernel[(N,)](logits, out, N, V, logits.stride(0), BLOCK_SIZE=BS)
    return out
