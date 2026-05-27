"""Worked solutions for Chapter 10 exercises."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: plain batched matmul (no masks; assumes divisibility).
# ---------------------------------------------------------------------------
@triton.jit
def batched_matmul_nomask_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0); pid_m = tl.program_id(1); pid_n = tl.program_id(2)
    a_ptr += pid_b * stride_ab
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K // BLOCK_K):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Exercise 2: linear without bias.  Y = X @ W^T
# ---------------------------------------------------------------------------
@triton.jit
def linear_nobias_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Exercise 3: linear with bias, no activation.  Y = X @ W^T + b
# ---------------------------------------------------------------------------
@triton.jit
def linear_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Exercise 4: see src/ch10_batched_linear/linear_bias_gelu.py for the
# reference implementation. Reimplementing it from scratch is the point of
# the exercise; the production version is the answer key.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 5: fused linear + bias + residual.  Y = X @ W^T + b + R
# ---------------------------------------------------------------------------
@triton.jit
def linear_bias_residual_kernel(
    x_ptr, w_ptr, b_ptr, r_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_rm, stride_rn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    r_ptrs = r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    rmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    r = tl.load(r_ptrs, mask=rmask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :] + r

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=rmask)


# ---------------------------------------------------------------------------
# Exercise 6 (prose): why fused linear + LayerNorm is hard
# ---------------------------------------------------------------------------
# LayerNorm normalizes across the *last* dim of its input. For a linear
# layer output Y = X @ W^T + b of shape (M, N), LayerNorm needs the mean
# and variance over the full N for each of the M rows.
#
# A matmul program owns a (BLOCK_M, BLOCK_N) tile -- it only sees
# BLOCK_N columns of any given row. So a single program cannot compute
# the row-wise mean and variance without seeing the other programs'
# outputs.
#
# Options:
#   1. Two-pass: first launch the matmul (writing Y to DRAM), then
#      launch a LayerNorm. This is what nn.Linear -> nn.LayerNorm does.
#      You pay one round-trip of Y through DRAM, which is the cost the
#      fusion was supposed to save.
#   2. Cooperative reduction: have the matmul programs that cover a
#      single row write to a partial-sum buffer, atomically add their
#      contribution to a per-row sum and sum-of-squares, then a second
#      pass reads those statistics and does the normalize+store. Works
#      but adds atomics and a second pass anyway.
#   3. Streaming if BLOCK_N == N: when one program owns an entire row
#      (only possible for small N), it can compute the row stats from
#      its own accumulator. This is the FlashAttention trick applied
#      to LayerNorm; see Chapter 18 for residual+RMSNorm where this
#      pattern actually fits.
