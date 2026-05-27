"""ALiBi bias kernel.

ALiBi adds a per-head linear bias `-m_h * |i - j|` to each entry of the
attention score matrix before softmax. Materializing a [H, S, S] bias is only
useful for testing and prototyping. Production fuses the term into the
attention kernel; you compute `-m_h * |i - j|` on the fly per tile.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def build_alibi_slopes(num_heads: int, device: str | torch.device = "cuda") -> torch.Tensor:
    """Standard ALiBi slopes: geometric sequence, head h has slope 2^(-8h/H).

    Returns a [H] fp32 tensor.
    """
    # The original ALiBi paper handles non-power-of-two head counts with an
    # interpolation trick; we implement the common power-of-two case.
    powers = torch.arange(1, num_heads + 1, device=device, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * powers / num_heads)


@triton.jit
def _alibi_bias_kernel(
    slopes_ptr, out_ptr,
    H, S,
    stride_oh, stride_oi, stride_oj,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """out[h, i, j] = -slopes[h] * |i - j|."""
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    slope = tl.load(slopes_ptr + pid_h)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    i_mask = offs_i < S
    j_mask = offs_j < S

    diff = offs_i[:, None] - offs_j[None, :]
    abs_diff = tl.where(diff < 0, -diff, diff).to(tl.float32)
    bias = -slope * abs_diff

    out_ptrs = (
        out_ptr + pid_h * stride_oh
        + offs_i[:, None] * stride_oi + offs_j[None, :] * stride_oj
    )
    tl.store(out_ptrs, bias, mask=i_mask[:, None] & j_mask[None, :])


def alibi_bias(
    num_heads: int,
    seq_len: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    block_i: int = 32,
    block_j: int = 32,
) -> torch.Tensor:
    """Materialize the full [H, S, S] ALiBi bias. Useful for testing only."""
    slopes = build_alibi_slopes(num_heads, device=device)
    out = torch.empty(num_heads, seq_len, seq_len, device=device, dtype=dtype)
    grid = (num_heads, triton.cdiv(seq_len, block_i), triton.cdiv(seq_len, block_j))
    _alibi_bias_kernel[grid](
        slopes, out,
        num_heads, seq_len,
        *out.stride(),
        BLOCK_I=block_i, BLOCK_J=block_j,
    )
    return out
