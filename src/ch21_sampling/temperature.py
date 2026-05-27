"""Temperature scaling kernel. See Chapter 21."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _temperature_kernel(
    x_ptr, y_ptr, inv_t,
    N, V, stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * stride_row + cols, x * inv_t, mask=mask)


def temperature_scale(logits, temperature):
    """y = logits / temperature, per row."""
    assert logits.is_cuda
    N, V = logits.shape
    y = torch.empty_like(logits)
    inv_t = 1.0 / float(temperature)
    BS = triton.next_power_of_2(V)
    _temperature_kernel[(N,)](
        logits, y, inv_t, N, V, logits.stride(0), BLOCK_SIZE=BS
    )
    return y
