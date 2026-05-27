"""Chapter 8 worked solutions."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# Exercise 1 & 2 ship in src/; see gelu.py and silu.py. Below: bias+gelu.

# Exercise 3: fused bias + GELU -------------------------------------------

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, y_ptr, N, D,
                        BLOCK_SIZE: tl.constexpr,
                        INV_SQRT2: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_idx = offs % D
    b = tl.load(b_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    z = x + b
    y = 0.5 * z * (1.0 + tl.erf(z * INV_SQRT2))
    tl.store(y_ptr + offs, y, mask=mask)


def bias_gelu(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """x: (..., D), b: (D,). Returns gelu(x + b)."""
    assert b.ndim == 1 and x.shape[-1] == b.shape[0]
    orig = x.shape
    D = b.shape[0]
    x_flat = x.contiguous().reshape(-1)
    N = x_flat.numel()
    y = torch.empty_like(x_flat, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _bias_gelu_kernel[grid](x_flat, b, y, N, D, BLOCK_SIZE=BLOCK, INV_SQRT2=_INV_SQRT2, num_warps=4)
    return y.reshape(orig).to(x.dtype)


# Exercise 4: SwiGLU ships in src/swiglu.py.


# Exercise 5: skeleton for linear+bias+activation epilogue ----------------

@triton.jit
def _linear_bias_gelu_skeleton_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Skeleton only. The matmul body is Chapter 10; here we wire up the
    epilogue (bias add + GELU) so the reader can see where the activation
    slots in.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # TODO: matmul accumulator goes here (see Chapter 10).
    # for k in range(0, K, BLOCK_K):
    #     a = tl.load(...); b = tl.load(...); acc += tl.dot(a, b)

    # ---- Epilogue: bias + GELU ----
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]
    acc = 0.5 * acc * (1.0 + tl.erf(acc * _INV_SQRT2))

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(y_ptrs, acc, mask=mask)


# Exercise 6: GEGLU ships in src/swiglu.py (alongside SwiGLU).


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    b = torch.randn(16, device="cuda", dtype=torch.float32)
    out = bias_gelu(x, b)
    ref = torch.nn.functional.gelu(x + b, approximate="none")
    print("bias_gelu max diff:", (out - ref).abs().max().item())
