"""Per-channel int8 dequantization.

Given an int8 weight matrix Q of shape (M, N) and an fp16/fp32 scale vector
of shape (M,), produce an fp16 matrix W where W[m, n] = scale[m] * Q[m, n].

The kernel is memory-bound. Each program handles a (BLOCK_M, BLOCK_N) tile.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_int8_per_channel_kernel(
    q_ptr,           # *int8     (M, N)
    scale_ptr,       # *fp16/fp32(M,)
    out_ptr,         # *fp16     (M, N)
    M, N,
    stride_qm, stride_qn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0)
    # q is int8; cast to fp32 for the multiply, then down to fp16 at store time.
    q_f = q.to(tl.float32)

    s = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    out = (s[:, None] * q_f).to(tl.float16)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def dequant_int8_per_channel(
    q: torch.Tensor,
    scale: torch.Tensor,
    block_m: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    """Dequantize an int8 matrix with a per-row fp scale into fp16.

    Args:
        q: int8 tensor, shape (M, N), CUDA.
        scale: fp16 or fp32 tensor, shape (M,), CUDA. One scale per row.

    Returns:
        fp16 tensor, shape (M, N).
    """
    assert q.is_cuda and scale.is_cuda
    assert q.dtype == torch.int8
    assert scale.dtype in (torch.float16, torch.float32)
    assert q.dim() == 2 and scale.dim() == 1
    M, N = q.shape
    assert scale.shape[0] == M

    out = torch.empty((M, N), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _dequant_int8_per_channel_kernel[grid](
        q, scale, out,
        M, N,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
