"""Fused dequant + bias + GELU.

Reads int8 q with per-row scale, loads an fp16 bias of shape (N,), adds the
bias and applies tanh-approx GELU in one pass. Pure fusion exercise —
everything stays in registers from load through activation to store.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_bias_gelu_kernel(
    q_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
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

    s = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # dequant in fp32 to keep the activation accurate
    x = s[:, None] * q.to(tl.float32) + b[None, :]

    # tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    k0 = 0.7978845608028654    # sqrt(2/pi)
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def dequant_bias_gelu_fused(
    q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    block_m: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    """Fused dequant + bias add + GELU. Returns fp16.

    q:     int8 (M, N)
    scale: fp16/fp32 (M,)
    bias:  fp16/fp32 (N,)
    """
    assert q.is_cuda and scale.is_cuda and bias.is_cuda
    assert q.dtype == torch.int8
    M, N = q.shape
    assert scale.shape == (M,) and bias.shape == (N,)
    out = torch.empty((M, N), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _dequant_bias_gelu_kernel[grid](
        q, scale, bias, out,
        M, N,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
