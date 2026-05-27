"""1-D copy kernel.

The simplest possible non-trivial Triton kernel: read a tile, write the tile.
Useful as a peak-bandwidth sanity check for the GPU you happen to be on, and
as a structural template that the rest of the chapter builds on.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def copy(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous(), "copy_kernel expects a contiguous CUDA tensor"
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    copy_kernel[grid](x, out, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out
