"""GELU activation: exact and tanh approximation."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# 1/sqrt(2) for the exact form, and the tanh-approx constants.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_COEFF = 0.044715


@triton.jit
def _gelu_kernel(
    x_ptr, y_ptr, N,
    BLOCK_SIZE: tl.constexpr,
    APPROX: tl.constexpr,        # 0 = exact (erf), 1 = tanh
    INV_SQRT2: tl.constexpr,
    SQRT_2_OVER_PI: tl.constexpr,
    COEFF: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if APPROX == 0:
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        y = 0.5 * x * (1.0 + tl.erf(x * INV_SQRT2))
    else:
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        # tl.tanh exists in recent Triton; fall back via sigmoid identity if not.
        y = 0.5 * x * (1.0 + 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0)
    # Cast back to the input dtype on store.
    in_dtype = tl.load(x_ptr + offs, mask=mask, other=0.0).dtype
    tl.store(y_ptr + offs, y.to(in_dtype), mask=mask)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """GELU. `approximate` is "none" (exact, via erf) or "tanh"."""
    assert x.is_cuda
    assert approximate in ("none", "tanh")
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _gelu_kernel[grid](
        x, y, N,
        BLOCK_SIZE=BLOCK_SIZE,
        APPROX=(0 if approximate == "none" else 1),
        INV_SQRT2=_INV_SQRT2,
        SQRT_2_OVER_PI=_SQRT_2_OVER_PI,
        COEFF=_COEFF,
        num_warps=4,
    )
    return y
