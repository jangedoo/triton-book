"""Fused residual + LayerNorm + dropout. The Chapter 30 worked example.

Computes:

    h  = x + residual                # also returned, for the next layer
    n  = (h - mean(h)) * rsqrt(var(h) + eps) * weight + bias
    m  = bernoulli(1 - p) / (1 - p)
    y  = n * m

All in one pass. The mean and variance are computed in fp32 from the
fp16/bf16 input. The dropout mask is generated from `tl.rand` keyed by
the per-row pid plus a Python-side `seed`, so the kernel is reproducible
given a seed.

This is the "design your own kernel" exemplar for the playbook chapter.
The kernel is not a particularly hot path in modern LLMs (RMSNorm is more
common, dropout often gets disabled), but it shows every step of the
recipe: residual fusion, multi-statistic reduction, RNG, dtype policy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_ln_dropout_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    b_ptr,
    h_ptr,
    y_ptr,
    stride_xm,
    stride_rm,
    stride_hm,
    stride_ym,
    N,
    eps,
    p_keep,
    seed,
    BLOCK_SIZE: tl.constexpr,
    APPLY_DROPOUT: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * stride_rm + offs, mask=mask, other=0.0).to(tl.float32)
    h = x + r

    # Welford in one pass is fancier than we need for educational purposes;
    # the textbook two-pass mean+var is fine for typical hidden dims.
    mean = tl.sum(h, axis=0) / N
    centered = tl.where(mask, h - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    n = centered * inv * w + b

    if APPLY_DROPOUT:
        rand = tl.rand(seed, row * BLOCK_SIZE + offs)
        keep = (rand < p_keep).to(tl.float32)
        n = n * keep / p_keep

    tl.store(h_ptr + row * stride_hm + offs, h, mask=mask)
    tl.store(y_ptr + row * stride_ym + offs, n, mask=mask)


def fused_residual_ln_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    p: float = 0.0,
    eps: float = 1e-5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused `h = x + residual; y = dropout(LayerNorm(h), p)`.

    Returns `(h, y)`. `h` is the new residual stream; `y` is the
    normalized + optionally dropped value.

    Args:
        x, residual: (..., N) tensors, same shape, same dtype.
        weight, bias: (N,) tensors.
        p: dropout probability. If 0, the dropout step is compiled out.
        eps: LayerNorm epsilon.
        seed: RNG seed. Same seed produces the same mask.
    """
    if x.shape != residual.shape:
        raise ValueError("fused_residual_ln_dropout: shape mismatch")
    if not (0.0 <= p < 1.0):
        raise ValueError("fused_residual_ln_dropout: p must be in [0, 1)")

    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    r2 = residual.reshape(-1, N).contiguous()
    M = x2.shape[0]

    h = torch.empty_like(x2)
    y = torch.empty_like(x2)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _fused_residual_ln_dropout_kernel[(M,)](
        x2, r2, weight, bias, h, y,
        x2.stride(0), r2.stride(0), h.stride(0), y.stride(0),
        N, eps, 1.0 - p, seed,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_DROPOUT=(p > 0.0),
        num_warps=4 if BLOCK_SIZE <= 2048 else 8,
    )
    return h.reshape(x.shape), y.reshape(x.shape)
