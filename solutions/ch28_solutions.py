"""Worked solutions for Chapter 28 exercises."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "ch28_mini_lib"),
)


# Exercise 1: Add a `geglu` kernel.
@triton.jit
def _geglu_kernel(a_ptr, b_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh-approximate GELU
    k = 0.7978845608028654  # sqrt(2/pi)
    gelu = 0.5 * a * (1.0 + tl.math.tanh(k * (a + 0.044715 * a * a * a)))
    tl.store(y_ptr + offs, gelu * b, mask=mask)


def geglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    y = torch.empty_like(a_flat)
    N = a_flat.numel()
    BLOCK_SIZE = 1024
    _geglu_kernel[(triton.cdiv(N, BLOCK_SIZE),)](a_flat, b_flat, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view_as(a)


# Exercise 2: smoke import.
def smoke_import() -> None:
    import mini_triton_llm as mtl

    for name in [
        "rmsnorm",
        "residual_rmsnorm",
        "softmax",
        "rope",
        "swiglu",
        "flash_attention",
        "cross_entropy",
    ]:
        assert hasattr(mtl, name), name
    print("ok")


# Exercise 3: bench two kernels.
def bench_two() -> None:
    from mini_triton_llm import rmsnorm, softmax
    from mini_triton_llm.benchmarking import compare, print_table

    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")
    rows = [
        compare(
            "rmsnorm",
            lambda: rmsnorm(x, w),
            lambda: F.rms_norm(x, (x.shape[-1],), w),
        ),
        compare(
            "softmax",
            lambda: softmax(x),
            lambda: torch.softmax(x.float(), dim=-1).half(),
        ),
    ]
    print_table(rows)


# Exercise 4: type-hint discipline. (See the modules themselves; this is
# a stylistic exercise rather than a runnable one.)


# Exercise 5: kernels_available().
def kernels_available() -> list[str]:
    import mini_triton_llm as mtl

    return list(mtl.__all__)


# Exercise 6: autograd-wrapped rmsnorm. We implement dx and dw in PyTorch
# for clarity here; a Triton-native backward kernel is left to the reader.
class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        from mini_triton_llm import rmsnorm

        y = rmsnorm(x, weight, eps=eps)
        # Save fp32 inv_rms for an accurate backward.
        var = (x.float() ** 2).mean(-1, keepdim=True)
        inv = torch.rsqrt(var + eps)
        ctx.save_for_backward(x, weight, inv)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, inv = ctx.saved_tensors
        x_f = x.float()
        dy_f = (dy * weight).float()
        N = x.shape[-1]
        # dy/dx = inv * (dy_f - (1/N) * x * inv^2 * sum(dy_f * x))
        s = (dy_f * x_f).sum(-1, keepdim=True)
        dx = inv * (dy_f - (x_f * inv * inv * s) / N)
        dx = dx.to(x.dtype)
        # dw = sum over batch of dy * x * inv
        dw = (dy.float() * x_f * inv).reshape(-1, N).sum(0).to(weight.dtype)
        return dx, dw, None


def rmsnorm_autograd(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return RMSNormFn.apply(x, w, eps)


# Notes on Exercise 6: against `nn.RMSNorm`, the forward wins on memory
# bandwidth (one kernel launch, fp32 accumulate). The backward as written
# here is a PyTorch fallback; a Triton-native backward would need the same
# row-reduction pattern as the forward, plus a second pass for dw.


if __name__ == "__main__":
    smoke_import()
