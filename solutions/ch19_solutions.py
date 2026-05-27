"""Solutions for Chapter 19 exercises."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Exercise 1 (Beginner): bias + SiLU only.
@triton.jit
def _silu_bias_kernel(
    x_ptr, b_ptr, y_ptr,
    stride_x_m, stride_x_h, stride_y_m, stride_y_h,
    M, H,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H
    mask = mask_m[:, None] & mask_h[None, :]
    x = tl.load(x_ptr + offs_m[:, None] * stride_x_m + offs_h[None, :] * stride_x_h,
                mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    z = x + b[None, :]
    y = z * tl.sigmoid(z)
    tl.store(y_ptr + offs_m[:, None] * stride_y_m + offs_h[None, :] * stride_y_h,
             y, mask=mask)


def silu_bias(x, b, BM=32, BH=128):
    M, H = x.shape
    y = torch.empty_like(x)
    grid = (triton.cdiv(M, BM), triton.cdiv(H, BH))
    _silu_bias_kernel[grid](x, b, y, x.stride(0), x.stride(1),
                            y.stride(0), y.stride(1), M, H,
                            BLOCK_M=BM, BLOCK_H=BH)
    return y


# Exercise 2 (Beginner): use the chapter kernel with both biases as None.
from ch19_fused_swiglu import swiglu_bias

def swiglu_no_bias(xg, xu):
    return swiglu_bias(xg, None, xu, None)


# Exercise 3 (Beginner): GEGLU via the dedicated kernel.
from ch19_fused_swiglu import geglu_bias

def geglu_demo(xg, bg, xu, bu):
    return geglu_bias(xg, bg, xu, bu)


# Intermediate 1: matrix of optional biases.
def swiglu_any_bias(xg, bg, xu, bu):
    return swiglu_bias(xg, bg, xu, bu)  # chapter kernel already handles all four cases


# Intermediate 2: in-place writeback into x_up.
@triton.jit
def _swiglu_inplace_kernel(
    xg_ptr, bg_ptr, xu_ptr, bu_ptr,
    stride_xg_m, stride_xg_h, stride_xu_m, stride_xu_h,
    M, H,
    HAS_BG: tl.constexpr, HAS_BU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H
    mask = mask_m[:, None] & mask_h[None, :]
    xg = tl.load(xg_ptr + offs_m[:, None] * stride_xg_m + offs_h[None, :] * stride_xg_h,
                 mask=mask, other=0.0).to(tl.float32)
    xu = tl.load(xu_ptr + offs_m[:, None] * stride_xu_m + offs_h[None, :] * stride_xu_h,
                 mask=mask, other=0.0).to(tl.float32)
    if HAS_BG:
        bg = tl.load(bg_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        xg = xg + bg[None, :]
    if HAS_BU:
        bu = tl.load(bu_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        xu = xu + bu[None, :]
    y = xg * tl.sigmoid(xg) * xu
    # Write back into x_up — its data is already in registers.
    tl.store(xu_ptr + offs_m[:, None] * stride_xu_m + offs_h[None, :] * stride_xu_h,
             y, mask=mask)


def swiglu_inplace(xg, bg, xu, bu, BM=32, BH=128):
    M, H = xg.shape
    grid = (triton.cdiv(M, BM), triton.cdiv(H, BH))
    _bg = bg if bg is not None else xg
    _bu = bu if bu is not None else xu
    _swiglu_inplace_kernel[grid](
        xg, _bg, xu, _bu,
        xg.stride(0), xg.stride(1), xu.stride(0), xu.stride(1),
        M, H,
        HAS_BG=bg is not None, HAS_BU=bu is not None,
        BLOCK_M=BM, BLOCK_H=BH,
    )
    return xu  # written in place


# Advanced: backward sketch for SwiGLU.
# y = silu(g) * u, where g = xg + bg, u = xu + bu.
#   sig = sigmoid(g)
#   silu = g * sig
#   d_silu_dg = sig + g * sig * (1 - sig) = sig * (1 + g * (1 - sig))
#                                          = silu + sig * (1 - silu)
#   dy/dg = u * d_silu_dg
#   dy/du = silu
# So:
#   d xg = dy * u * d_silu_dg
#   d xu = dy * silu
#   d bg = sum_m(d xg)
#   d bu = sum_m(d xu)
@triton.jit
def _swiglu_backward_kernel(
    xg_ptr, bg_ptr, xu_ptr, bu_ptr, dy_ptr,
    dxg_ptr, dxu_ptr, dbg_ptr, dbu_ptr,
    stride_m, stride_h,
    M, H,
    HAS_BG: tl.constexpr, HAS_BU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H
    mask = mask_m[:, None] & mask_h[None, :]

    xg = tl.load(xg_ptr + offs_m[:, None] * stride_m + offs_h[None, :] * stride_h,
                 mask=mask, other=0.0).to(tl.float32)
    xu = tl.load(xu_ptr + offs_m[:, None] * stride_m + offs_h[None, :] * stride_h,
                 mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs_m[:, None] * stride_m + offs_h[None, :] * stride_h,
                 mask=mask, other=0.0).to(tl.float32)
    if HAS_BG:
        bg = tl.load(bg_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        g = xg + bg[None, :]
    else:
        g = xg
    if HAS_BU:
        bu = tl.load(bu_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        u = xu + bu[None, :]
    else:
        u = xu

    sig = tl.sigmoid(g)
    silu = g * sig
    d_silu_dg = sig + g * sig * (1.0 - sig)
    dxg = dy * u * d_silu_dg
    dxu = dy * silu

    tl.store(dxg_ptr + offs_m[:, None] * stride_m + offs_h[None, :] * stride_h,
             dxg, mask=mask)
    tl.store(dxu_ptr + offs_m[:, None] * stride_m + offs_h[None, :] * stride_h,
             dxu, mask=mask)
    if HAS_BG:
        # atomic add along the row dim
        col_sum_g = tl.sum(tl.where(mask, dxg, 0.0), axis=0)
        tl.atomic_add(dbg_ptr + offs_h, col_sum_g, mask=mask_h)
    if HAS_BU:
        col_sum_u = tl.sum(tl.where(mask, dxu, 0.0), axis=0)
        tl.atomic_add(dbu_ptr + offs_h, col_sum_u, mask=mask_h)
