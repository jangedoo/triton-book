"""W8A16 matmul: fp16 activations, int8 weights, fp32 accumulator, fp16 output.

Computes Y = X @ dequant(W) where:
    X:     (M, K)  fp16
    W:     (K, N)  int8          — note: K is the reduction axis
    scale: (N,)    fp16          — one scale per output channel
    Y:     (M, N)  fp16

The dequant happens on the fly inside the K-loop. For each (BLOCK_K, BLOCK_N)
weight tile loaded as int8, we cast to fp16, multiply by the per-column scale,
then feed into tl.dot. This is the W8A16 pattern that powers a large fraction
of consumer-GPU LLM inference today.

Notes:
    The scale is per-output-channel (per column of W) here. Many libraries
    instead store W as (N, K) row-major with a per-row scale; the math is
    identical, only the layout changes. We pick (K, N) for symmetry with
    the matmul chapters.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_matmul_kernel(
    x_ptr,           # *fp16  (M, K)
    w_ptr,           # *int8  (K, N)
    scale_ptr,       # *fp16  (N,)
    y_ptr,           # *fp16  (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Per-column scale — one scalar per column of W, broadcast along K.
    mask_n = offs_n < N
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)               # fp16
        w_q    = tl.load(w_ptrs, mask=w_mask, other=0)                  # int8

        # Dequant on the fly: cast int8 -> fp16, multiply by per-col scale.
        # s broadcasts across the K dim of the tile.
        w_fp = w_q.to(tl.float16) * s[None, :]                          # fp16

        acc += tl.dot(x_tile, w_fp)                                     # fp32 acc

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


def w8a16_matmul(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scale: torch.Tensor,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
) -> torch.Tensor:
    """fp16 activations x int8 weights → fp16 output, fp32 accumulator.

    Args:
        x:       (M, K) fp16
        w_int8:  (K, N) int8
        scale:   (N,)   fp16 — one scale per output channel

    Returns:
        y:       (M, N) fp16
    """
    assert x.is_cuda and w_int8.is_cuda and scale.is_cuda
    assert x.dtype == torch.float16
    assert w_int8.dtype == torch.int8
    assert scale.dtype == torch.float16
    M, K = x.shape
    K2, N = w_int8.shape
    assert K == K2
    assert scale.shape == (N,)

    y = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _w8a16_matmul_kernel[grid](
        x, w_int8, scale, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return y
