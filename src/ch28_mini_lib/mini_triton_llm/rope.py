"""Rotary positional embedding (LLaMA-style, non-interleaved).

Splits the last dim D in half. For each pair (x[..., i], x[..., i+D/2]):

    x'[i]      = x[i]      * cos[m, i] - x[i+D/2] * sin[m, i]
    x'[i+D/2]  = x[i+D/2]  * cos[m, i] + x[i]     * sin[m, i]

where `m = position_offset + token_index`. Lifted from Chapter 16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    stride_cs, stride_cd,
    H,
    D_HALF: tl.constexpr,
    offset,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    offs = tl.arange(0, D_HALF)
    pos = s + offset

    x_base = x_ptr + b * stride_xb + s * stride_xs + h * stride_xh
    y_base = y_ptr + b * stride_yb + s * stride_ys + h * stride_yh

    x0 = tl.load(x_base + offs * stride_xd).to(tl.float32)
    x1 = tl.load(x_base + (offs + D_HALF) * stride_xd).to(tl.float32)

    cos = tl.load(cos_ptr + pos * stride_cs + offs * stride_cd).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_cs + offs * stride_cd).to(tl.float32)

    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin

    tl.store(y_base + offs * stride_yd, y0)
    tl.store(y_base + (offs + D_HALF) * stride_yd, y1)


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings to `x`.

    Args:
        x:   (B, S, H, D) with D even. fp16/bf16/fp32.
        cos: (S_max, D/2). Same dtype family.
        sin: (S_max, D/2).
        offset: position offset for the first token in `x` (use for KV-cache
            decode where the new token is at position `past_len`).

    Returns:
        Same shape and dtype as `x`.
    """
    if x.dim() != 4:
        raise ValueError("rope: x must be (B, S, H, D)")
    B, S, H, D = x.shape
    if D % 2 != 0:
        raise ValueError("rope: head_dim must be even")

    y = torch.empty_like(x)
    grid = (B, S, H)
    _rope_kernel[grid](
        x, cos, sin, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        cos.stride(0), cos.stride(1),
        H,
        D_HALF=D // 2,
        offset=offset,
    )
    return y
