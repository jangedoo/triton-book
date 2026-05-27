"""Fused linear + bias + GELU.

Y = gelu(X @ W^T + b)
  X: (B*T, in)        treat the leading batch+token axes as one
  W: (out, in)        PyTorch nn.Linear convention -- W is (out, in)
  b: (out,)
  Y: (B*T, out)

The transpose on W is baked into the pointer math: stride_w_out is the
"row stride" (out axis), stride_w_in is the "k stride" (in axis). No
physical transpose. We treat W as if it were (in, out) by walking
stride_w_in down rows and stride_w_out across columns.

Epilogue does bias broadcast and GELU (tanh approximation) inside the
kernel, so the intermediate `X @ W^T + b` never round-trips through
DRAM. That is the entire performance story.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def linear_bias_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = B*T, N = out, K = in
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # X tile: (BLOCK_M, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # W tile: we want a (BLOCK_K, BLOCK_N) slice of W^T, which is the
    # same as a (BLOCK_N, BLOCK_K) slice of W, transposed. We get the
    # transpose for free by addressing W with offs_n on the "row" and
    # offs_k on the "col", then loading as (BLOCK_K, BLOCK_N) via the
    # broadcast pattern below.
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        w_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Bias broadcast over the M axis.
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # GELU (tanh approximation).
    c0 = 0.7978845608028654   # sqrt(2/pi)
    inner = c0 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    y = acc.to(y_ptr.dtype.element_ty)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


def linear_bias_gelu(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Y = gelu(X @ W^T + b).

    x : (..., in)         any leading shape, will be flattened
    w : (out, in)
    b : (out,)
    Returns y of shape x.shape[:-1] + (out,).
    """
    assert x.is_cuda and w.is_cuda and bias.is_cuda
    assert x.dtype == w.dtype == bias.dtype

    out_features, in_features = w.shape
    assert x.shape[-1] == in_features
    assert bias.shape == (out_features,)

    leading = x.shape[:-1]
    x_2d = x.reshape(-1, in_features)
    M, K = x_2d.shape
    N = out_features

    y_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    linear_bias_gelu_kernel[grid](
        x_2d, w, bias, y_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y_2d.reshape(*leading, N)
