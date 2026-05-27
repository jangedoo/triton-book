"""Persistent matmul skeleton -- READ THE HARDWARE NOTE.

================================================================
HARDWARE REQUIREMENT
================================================================
This kernel is untested on Turing (sm_75). Persistent matmul gives a
real win on Ampere (sm_80) and newer where tensor-core throughput, L2
size, and SM count are large enough that launch and scheduling overhead
become measurable.

On a 2070 SUPER (sm_75) the speedup over the Chapter 9 grouped matmul
will be small to negative because:
  - sm_75 tl.dot is on first-gen tensor cores; arithmetic intensity is
    already capped well below the launch-overhead regime.
  - 40 SMs leaves less headroom for one-program-per-SM scheduling.

Run this on an A100 / H100 / RTX 40xx if you want to see the benefit.
================================================================

The design:
  - Launch NUM_SMS programs (or thereabouts), not one program per output tile.
  - Each program loops over a strided slice of the (pid_m, pid_n) tile space.
  - Inside the loop, do the same K-loop matmul as Chapter 9.
  - Reuse the grouped-ordering trick to keep L2 reuse on the row-stripes.

Refer to the official Triton persistent matmul tutorial for the
production-ready version with software pipelining and TMA on Hopper.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- one program owns many output tiles ----------------------------
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    # Each program walks tiles `start_pid, start_pid + NUM_SMS, ...`
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        # Grouped-ordering mapping from tile_id to (pid_m, pid_n).
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            rem = K - k * BLOCK_K
            a = tl.load(a_ptrs,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem),
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N),
                        other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def persistent_matmul_skeleton(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    group_size_m: int = 8,
) -> torch.Tensor:
    """Launches the persistent matmul kernel. Untested on sm < 80."""
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    # One program per SM. Bigger or smaller may help -- benchmark.
    grid = (num_sms,)
    persistent_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=num_sms,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
    )
    return c
