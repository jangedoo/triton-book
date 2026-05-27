"""Naive row-major Triton matmul.

Each program computes one BLOCK_M x BLOCK_N tile of C by looping over K in
chunks of BLOCK_K. Accumulator is fp32 even for fp16 inputs. Program ids
are mapped to (pid_m, pid_n) in row-major order, which is the simplest
mapping but leaves L2 reuse on the table -- see grouped_matmul.py for the
super-grouped variant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_naive_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ---- program id -> output tile (row-major) -------------------------
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ---- offsets within the tile ---------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                     # (BLOCK_K,)

    # ---- starting pointers for this tile's A row-stripe and B col-stripe
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ---- accumulator in fp32 -------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- main K-loop ---------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # tl.dot uses tensor cores when shapes and dtypes allow.
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- cast back and store -------------------------------------------
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
) -> torch.Tensor:
    """Compute C = A @ B with the naive row-major launcher.

    a : (M, K), fp16/bf16/fp32
    b : (K, N), same dtype as a
    """
    assert a.is_cuda and b.is_cuda, "inputs must be on cuda"
    assert a.shape[1] == b.shape[0], "inner dims must match"
    assert a.dtype == b.dtype, "matching dtypes required"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_naive_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c
