"""Fused GEGLU MLP epilogue: gelu_tanh(x_gate + b_gate) * (x_up + b_up).

GELU variant of the SwiGLU kernel from swiglu_bias.py. See Chapter 19.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def geglu_bias_ref(x_gate, b_gate, x_up, b_up):
    g = x_gate if b_gate is None else x_gate + b_gate
    u = x_up if b_up is None else x_up + b_up
    return F.gelu(g.float(), approximate="tanh").to(g.dtype) * u


@triton.jit
def _gelu_tanh(x):
    k0 = 0.7978845608028654  # sqrt(2 / pi)
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    return 0.5 * x * (1.0 + tl.math.tanh(inner))


@triton.jit
def _geglu_bias_kernel(
    xg_ptr, bg_ptr, xu_ptr, bu_ptr, y_ptr,
    stride_xg_m, stride_xg_h,
    stride_xu_m, stride_xu_h,
    stride_y_m, stride_y_h,
    M, H,
    HAS_BG: tl.constexpr,
    HAS_BU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H
    mask = mask_m[:, None] & mask_h[None, :]

    xg = tl.load(
        xg_ptr + offs_m[:, None] * stride_xg_m + offs_h[None, :] * stride_xg_h,
        mask=mask, other=0.0,
    ).to(tl.float32)
    xu = tl.load(
        xu_ptr + offs_m[:, None] * stride_xu_m + offs_h[None, :] * stride_xu_h,
        mask=mask, other=0.0,
    ).to(tl.float32)

    if HAS_BG:
        bg = tl.load(bg_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        xg = xg + bg[None, :]
    if HAS_BU:
        bu = tl.load(bu_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        xu = xu + bu[None, :]

    gelu_g = _gelu_tanh(xg)
    y = gelu_g * xu

    tl.store(
        y_ptr + offs_m[:, None] * stride_y_m + offs_h[None, :] * stride_y_h,
        y, mask=mask,
    )


def geglu_bias(x_gate, b_gate, x_up, b_up, BLOCK_M=32, BLOCK_H=128):
    assert x_gate.shape == x_up.shape
    M, H = x_gate.shape
    y = torch.empty_like(x_gate)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))
    bg = b_gate if b_gate is not None else x_gate
    bu = b_up if b_up is not None else x_up
    _geglu_bias_kernel[grid](
        x_gate, bg, x_up, bu, y,
        x_gate.stride(0), x_gate.stride(1),
        x_up.stride(0), x_up.stride(1),
        y.stride(0), y.stride(1),
        M, H,
        HAS_BG=b_gate is not None,
        HAS_BU=b_up is not None,
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
    )
    return y
