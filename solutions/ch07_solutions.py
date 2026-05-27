"""Chapter 7 worked solutions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Exercise 2: optional weight ---------------------------------------------

@triton.jit
def _rms_optional_w_kernel(x_ptr, y_ptr, w_ptr, row_stride, H, eps,
                              BLOCK_SIZE: tl.constexpr, WEIGHTED: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(ms + eps)
    y = x * rstd
    if WEIGHTED:
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = y * w
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


def rmsnorm_optional_weight(x, weight=None, eps=1e-6):
    M, H = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(H)
    weighted = weight is not None
    if not weighted:
        weight = torch.empty(1, device=x.device, dtype=x.dtype)
    _rms_optional_w_kernel[(M,)](x, y, weight, x.stride(0), H, eps,
                                    BLOCK_SIZE=BLOCK, WEIGHTED=weighted, num_warps=4)
    return y


# Exercise 3: eps as constexpr -------------------------------------------

@triton.jit
def _rms_eps_constexpr_kernel(x_ptr, y_ptr, w_ptr, row_stride, H,
                                  BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(ms + EPS)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * row_stride + cols, (x * rstd * w), mask=mask)


def rmsnorm_eps_constexpr(x, w, eps=1e-6):
    M, H = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(H)
    _rms_eps_constexpr_kernel[(M,)](x, y, w, x.stride(0), H,
                                          BLOCK_SIZE=BLOCK, EPS=eps, num_warps=4)
    return y


# Exercise 5: in-place RMSNorm -------------------------------------------

def rmsnorm_inplace_(x, w, eps=1e-6):
    """Mutates x. The kernel reads then writes; the whole row is in regs
    after the load, so reading-then-writing the same memory is safe."""
    from src.ch07_rmsnorm.rmsnorm import _rmsnorm_fwd_kernel
    M, H = x.shape
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(H)
    _rmsnorm_fwd_kernel[(M,)](x, x, w, rstd, x.stride(0), x.stride(0), H, eps,
                                  BLOCK_SIZE=BLOCK, num_warps=4)
    return x


# Exercise 6: fused residual-add + RMSNorm --------------------------------

@triton.jit
def _residual_rms_kernel(x_ptr, r_ptr, y_ptr, sum_ptr, w_ptr,
                            row_stride, H, eps,
                            BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    # Save the post-add tensor for the next residual stream.
    in_dtype = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).dtype
    tl.store(sum_ptr + row * row_stride + cols, s.to(in_dtype), mask=mask)
    ms = tl.sum(s * s, axis=0) / H
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = s * rstd * w
    tl.store(y_ptr + row * row_stride + cols, y.to(in_dtype), mask=mask)


def residual_rmsnorm(x, residual, w, eps=1e-6):
    """Returns (y, x_plus_residual). x and residual must share a dtype and shape."""
    M, H = x.shape
    y = torch.empty_like(x)
    s = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _residual_rms_kernel[(M,)](x, residual, y, s, w, x.stride(0), H, eps,
                                   BLOCK_SIZE=BLOCK, num_warps=4)
    return y, s


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print("Solutions module imports cleanly.")
