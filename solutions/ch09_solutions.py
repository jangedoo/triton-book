"""Worked solutions for Chapter 9 exercises.

Each kernel reuses the structure of src/ch09_matmul/naive_matmul.py and
only changes the epilogue or the K loop. Run on cuda.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: matmul + bias
# ---------------------------------------------------------------------------
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_bias(a, b, bias):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    matmul_bias_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return c


# ---------------------------------------------------------------------------
# Exercise 2: matmul + ReLU epilogue
# ---------------------------------------------------------------------------
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    acc = tl.maximum(acc, 0.0)
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Exercise 3: transposed A support -- launcher trick, no kernel change.
# ---------------------------------------------------------------------------
def matmul_with_optional_at(a, b, a_transposed: bool):
    """If a_transposed, interpret `a` as (K, M) physically but treat it as
    (M, K) logically by swapping which stride is row vs k."""
    from src.ch09_matmul.naive_matmul import matmul_naive_kernel

    if a_transposed:
        K, M = a.shape
        stride_am, stride_ak = a.stride(1), a.stride(0)
    else:
        M, K = a.shape
        stride_am, stride_ak = a.stride(0), a.stride(1)
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    matmul_naive_kernel[grid](
        a, b, c, M, N, K,
        stride_am, stride_ak,
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return c


# ---------------------------------------------------------------------------
# Exercise 4: matmul + GELU (tanh approximation)
# ---------------------------------------------------------------------------
@triton.jit
def matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # tanh-approx GELU
    c0 = 0.7978845608028654   # sqrt(2/pi)
    inner = c0 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Exercise 5: split-K (sketch)
# ---------------------------------------------------------------------------
# Skinny matmul: M, N small, K large. We split K across SPLIT_K programs;
# each computes a partial product over a K-chunk and atomic-adds into C.
# C must be zero-initialized before launch.
@triton.jit
def matmul_split_k_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Each program walks SPLIT_K * BLOCK_K stride; total K-loop iterations
    # per program is K / (SPLIT_K * BLOCK_K).
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        cur_k = (pid_k + k * SPLIT_K) * BLOCK_K
        rem = K - cur_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (tl.arange(0, BLOCK_K)[None, :] < rem), other=0.0)
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_K)[:, None] < rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += SPLIT_K * BLOCK_K * stride_ak
        b_ptrs += SPLIT_K * BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Exercise 6: roll-your-own super-grouping
# ---------------------------------------------------------------------------
# This is the same mapping as src/ch09_matmul/grouped_matmul.py but written
# out longhand. The derivation: imagine an 8x8 grid of output tiles and
# GROUP_SIZE_M = 4. We want pids 0..31 to fill the top 4 rows column by
# column (so K-tiles of A for those rows stay in L2 across all N), then
# pids 32..63 fill the bottom 4 rows the same way. So:
#   group_id     = pid // (GROUP_SIZE_M * num_pid_n)
#   first_pid_m  = group_id * GROUP_SIZE_M
#   within_group = pid % (GROUP_SIZE_M * num_pid_n)
#   pid_m        = first_pid_m + (within_group % GROUP_SIZE_M)
#   pid_n        = within_group // GROUP_SIZE_M
# The `tl.minimum` clamp handles the case where the last group is shorter
# than GROUP_SIZE_M.
