"""RoPE kernels: non-interleaved (LLaMA) and interleaved (GPT-J)."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# cos/sin cache
# ---------------------------------------------------------------------------


def build_cos_sin_cache(
    max_seq: int,
    dim: int,
    base: float = 10000.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the standard RoPE cos/sin cache.

    Returns (cos, sin), each of shape [max_seq, dim // 2], in fp32 by default.
    Keep these in fp32 even when the model runs in fp16/bf16; sin/cos lose
    precision quickly at long sequence lengths.
    """
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2 / dim))
    pos = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]   # [S, D/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# ---------------------------------------------------------------------------
# Non-interleaved (LLaMA) kernel + launcher
# ---------------------------------------------------------------------------


@triton.jit
def _rope_noninterleaved_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cs, stride_cd,
    POS_OFFSET,
    HALF: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)
    s_mask = offs_s < S

    base_x = x_ptr + b * stride_xb + h * stride_xh
    x1_ptrs = base_x + offs_s[:, None] * stride_xs + offs_h[None, :] * stride_xd
    x2_ptrs = base_x + offs_s[:, None] * stride_xs + (HALF + offs_h)[None, :] * stride_xd

    pos = offs_s + POS_OFFSET
    cos_ptrs = cos_ptr + pos[:, None] * stride_cs + offs_h[None, :] * stride_cd
    sin_ptrs = sin_ptr + pos[:, None] * stride_cs + offs_h[None, :] * stride_cd

    x1  = tl.load(x1_ptrs,  mask=s_mask[:, None], other=0.0).to(tl.float32)
    x2  = tl.load(x2_ptrs,  mask=s_mask[:, None], other=0.0).to(tl.float32)
    cos = tl.load(cos_ptrs, mask=s_mask[:, None], other=0.0)
    sin = tl.load(sin_ptrs, mask=s_mask[:, None], other=0.0)

    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos

    base_o = out_ptr + b * stride_ob + h * stride_oh
    o1_ptrs = base_o + offs_s[:, None] * stride_os + offs_h[None, :] * stride_od
    o2_ptrs = base_o + offs_s[:, None] * stride_os + (HALF + offs_h)[None, :] * stride_od
    tl.store(o1_ptrs, rot1, mask=s_mask[:, None])
    tl.store(o2_ptrs, rot2, mask=s_mask[:, None])


def rope_noninterleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_offset: int = 0,
    out: torch.Tensor | None = None,
    block_s: int = 32,
) -> torch.Tensor:
    """Apply non-interleaved (LLaMA) RoPE to x.

    x:   [B, H, S, D], any float dtype.
    cos: [S_max, D/2], fp32 recommended.
    sin: [S_max, D/2], fp32 recommended.
    pos_offset: absolute position of x[..., 0, :]. Use 0 for prefill;
        use current_len for single-token decode.
    out: optional output buffer; defaults to a fresh tensor like x.
    Returns out.
    """
    assert x.is_cuda
    B, H, S, D = x.shape
    assert D % 2 == 0, "RoPE dim must be even"
    if out is None:
        out = torch.empty_like(x)
    half = D // 2

    grid = (B * H, triton.cdiv(S, block_s))
    _rope_noninterleaved_kernel[grid](
        x, cos, sin, out,
        B, H, S, D,
        *x.stride(), *out.stride(), *cos.stride(),
        pos_offset,
        HALF=half, BLOCK_S=block_s,
    )
    return out


# ---------------------------------------------------------------------------
# Interleaved (GPT-J) kernel + launcher
# ---------------------------------------------------------------------------


@triton.jit
def _rope_interleaved_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cs, stride_cd,
    POS_OFFSET,
    HALF: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)
    s_mask = offs_s < S

    base_x = x_ptr + b * stride_xb + h * stride_xh
    # even and odd dims interleaved
    xe_ptrs = base_x + offs_s[:, None] * stride_xs + (2 * offs_h)[None, :]     * stride_xd
    xo_ptrs = base_x + offs_s[:, None] * stride_xs + (2 * offs_h + 1)[None, :] * stride_xd

    pos = offs_s + POS_OFFSET
    cos_ptrs = cos_ptr + pos[:, None] * stride_cs + offs_h[None, :] * stride_cd
    sin_ptrs = sin_ptr + pos[:, None] * stride_cs + offs_h[None, :] * stride_cd

    xe  = tl.load(xe_ptrs,  mask=s_mask[:, None], other=0.0).to(tl.float32)
    xo  = tl.load(xo_ptrs,  mask=s_mask[:, None], other=0.0).to(tl.float32)
    cos = tl.load(cos_ptrs, mask=s_mask[:, None], other=0.0)
    sin = tl.load(sin_ptrs, mask=s_mask[:, None], other=0.0)

    rot_e = xe * cos - xo * sin
    rot_o = xe * sin + xo * cos

    base_o = out_ptr + b * stride_ob + h * stride_oh
    oe_ptrs = base_o + offs_s[:, None] * stride_os + (2 * offs_h)[None, :]     * stride_od
    oo_ptrs = base_o + offs_s[:, None] * stride_os + (2 * offs_h + 1)[None, :] * stride_od
    tl.store(oe_ptrs, rot_e, mask=s_mask[:, None])
    tl.store(oo_ptrs, rot_o, mask=s_mask[:, None])


def rope_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_offset: int = 0,
    out: torch.Tensor | None = None,
    block_s: int = 32,
) -> torch.Tensor:
    """Apply interleaved (GPT-J) RoPE to x. Cache is shared with the non-interleaved variant."""
    assert x.is_cuda
    B, H, S, D = x.shape
    assert D % 2 == 0
    if out is None:
        out = torch.empty_like(x)
    half = D // 2

    grid = (B * H, triton.cdiv(S, block_s))
    _rope_interleaved_kernel[grid](
        x, cos, sin, out,
        B, H, S, D,
        *x.stride(), *out.stride(), *cos.stride(),
        pos_offset,
        HALF=half, BLOCK_S=block_s,
    )
    return out
