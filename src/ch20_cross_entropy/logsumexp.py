"""Row-wise logsumexp kernel. See Chapter 20."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def logsumexp_ref(x):
    return torch.logsumexp(x.float(), dim=-1)


@triton.jit
def _logsumexp_kernel(
    x_ptr, out_ptr,
    stride_x_row,
    N, V,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    lse = m + tl.log(s)
    tl.store(out_ptr + row, lse)


def logsumexp(x):
    assert x.is_cuda
    N, V = x.shape
    out = torch.empty(N, device=x.device, dtype=torch.float32)
    BS = triton.next_power_of_2(V)
    _logsumexp_kernel[(N,)](x, out, x.stride(0), N, V, BLOCK_SIZE=BS)
    return out
