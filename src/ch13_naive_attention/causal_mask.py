"""In-place causal mask kernel for the [B, H, S, S] score tensor."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def causal_mask_kernel(
    s_ptr,
    BH, S,
    stride_b, stride_m, stride_n,
    NEG_INF,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Set s[b, m, n] = -inf for n > m, in-place, one [BLOCK_M, BLOCK_N] tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < S
    mask_n = offs_n < S
    in_bounds = mask_m[:, None] & mask_n[None, :]

    s_ptrs = s_ptr + pid_b * stride_b \
        + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Future positions: column index > row index.
    causal = offs_n[None, :] > offs_m[:, None]

    # Load current value, replace with -inf where causal mask fires.
    cur = tl.load(s_ptrs, mask=in_bounds, other=0.0)
    new = tl.where(causal, NEG_INF, cur)
    tl.store(s_ptrs, new, mask=in_bounds)


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Apply causal mask in-place on a [B, H, S, S] score tensor."""
    assert scores.is_cuda and scores.ndim == 4
    B, H, S, _ = scores.shape
    BH = B * H
    s_flat = scores.reshape(BH, S, S)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), triton.cdiv(S, BLOCK_N), BH)

    causal_mask_kernel[grid](
        s_flat,
        BH, S,
        s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
        float("-inf"),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return scores
