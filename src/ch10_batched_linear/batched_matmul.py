"""Batched matmul: one kernel, 3D grid, batch-stride.

Input shapes:
  A: (B, M, K)
  B: (B, K, N)
Output:
  C: (B, M, N)

Strategy: a 3D launch grid (batch, num_pid_m, num_pid_n). Each program
computes one (BLOCK_M, BLOCK_N) tile for one batch element. The batch
stride is hoisted out of the K-loop -- each program adds
`pid_b * batch_stride` to its base pointers once.

The host-loop alternative (call matmul B times) costs B kernel launches
and B sets of scheduling state. Modern Triton ships work to the GPU
faster, but you still pay the launch latency. The 3D-grid variant is the
right default for the typical LLM workload where B is in the tens or
hundreds.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Offset every base pointer by this batch's stride. From here on the
    # kernel looks like a single-batch matmul.
    a_ptr += pid_b * stride_ab
    b_ptr += pid_b * stride_bb
    c_ptr += pid_b * stride_cb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 3 and b.dim() == 3
    Ba, M, K = a.shape
    Bb, K2, N = b.shape
    assert Ba == Bb and K == K2
    assert a.dtype == b.dtype

    c = torch.empty((Ba, M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    grid = (Ba, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    batched_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
