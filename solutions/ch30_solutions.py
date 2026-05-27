"""Worked solutions for Chapter 30 exercises.

The Beginner exercises are largely prose: design walks rather than code.
The Intermediate and Advanced exercises ship runnable kernels. Where a
kernel is the answer, it lives in `src/ch30_playbook/` and we import it
here.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "ch28_mini_lib"))


# ===========================================================================
# Exercise 1: pick a layer to fuse. We pick `Linear + ReLU`.
# ===========================================================================
#
# Step 1 -- PyTorch reference:
#
#     def linear_relu(x, W, b):
#         return torch.relu(x @ W.T + b)
#
# Step 2 -- inputs/outputs:
#
#     x: (M, K) fp16
#     W: (N, K) fp16
#     b: (N,)   fp16
#     y: (M, N) fp16
#
# Step 3 -- reduction axis:
#
#     The K axis is the reduction (dot product). The M and N axes are
#     mapped over independently. This is a matmul with a fused
#     elementwise activation on the output.


# ===========================================================================
# Exercise 2: grouped-query attention shapes.
# ===========================================================================
#
# Standard MHA:
#     Q: (B, H_q,  S, D)
#     K: (B, H_q,  S, D)   # same H
#     V: (B, H_q,  S, D)
#
# GQA with H_q=8, H_kv=2 (group_size=4):
#     Q: (B, H_q,  S, D)   # H_q = 8
#     K: (B, H_kv, S, D)   # H_kv = 2
#     V: (B, H_kv, S, D)
#
# Each KV head serves group_size=4 query heads. In the FlashAttention
# kernel, replace the head index used to load K, V with
# `kv_head = q_head // group_size`. Q stays addressed by q_head.
#
# The grid is still (cdiv(S, BLOCK_M), B * H_q). Only the K, V address
# computations change.


# ===========================================================================
# Exercise 3: sliding-window attention sketch.
# ===========================================================================
#
# Goal: query at position i attends to keys in [max(0, i-W), i].
#
# Block mapping (step 4): same as FlashAttention forward -- one program
# per (batch, head, query block).
#
# Loads (step 5): the inner loop iterates over key blocks. For a query
# block whose row range is [m_lo, m_hi), the relevant key block range is
# [max(0, m_lo - W) // BLOCK_N, m_hi // BLOCK_N + 1).
#
# Computes (step 6): same online softmax, with an extra mask for
# `k_idx < q_idx - W` set to -inf.
#
# Stores (step 7): same as FlashAttention.


# ===========================================================================
# Exercise 4: implement linear_relu (kernel) -- Intermediate.
# ===========================================================================
@triton.jit
def _linear_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]
    acc = tl.where(acc > 0, acc, 0.0)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def linear_relu(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N, K2 = W.shape
    assert K == K2
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_relu_kernel[grid](
        x, W, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y


# ===========================================================================
# Exercise 5: sliding-window attention -- left as a structural change to
# the Chapter 28 attention kernel. Sketch:
# ===========================================================================
#
# Inside _flash_attn_fwd, replace:
#     for start_n in range(0, n_end, BLOCK_N):
# with:
#     n_lo = max(0, start_m * BLOCK_M - WINDOW)
#     n_lo = (n_lo // BLOCK_N) * BLOCK_N
#     for start_n in range(n_lo, n_end, BLOCK_N):
#
# And add to the per-element causal mask:
#     window = offs_m[:, None] - n_idx[None, :] < WINDOW
#     qk = tl.where(causal & window, qk, -float("inf"))
#
# Test by comparing against a PyTorch reference that materializes the
# full [S, S] mask and zeroes out positions outside the window.


# ===========================================================================
# Exercise 6: autotune the linear_relu kernel.
# ===========================================================================
_CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _linear_relu_autotuned(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Body identical to _linear_relu_kernel; omitted for brevity. Copy
    # the loop body from above when you wire this in for real.
    pass


# Roofline note: at M=N=K=4096, fp16, this kernel is matmul-bound. The
# arithmetic intensity is ~K/2 = 2048 flop/byte, well above any GPU's
# ridge. Optimize for compute, not bandwidth: maximize tensor-core
# utilization, prefer larger BLOCK_M/BLOCK_N, watch register pressure.


if __name__ == "__main__":
    if torch.cuda.is_available():
        M, K, N = 256, 128, 64
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W = torch.randn(N, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, dtype=torch.float16, device="cuda")
        y = linear_relu(x, W, b)
        ref = F.relu(x @ W.T + b)
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
        print("linear_relu ok")
