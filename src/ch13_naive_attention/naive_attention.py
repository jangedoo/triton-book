"""Naive attention forward: glue around four sequential Triton kernels.

This deliberately materializes the [B, H, S, S] score and probability
tensors in HBM. Do not use this in production. It exists so the Chapter
14 FlashAttention forward has a baseline that shows the cost of the
quadratic intermediate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .qkt_matmul import qkt_matmul
from .causal_mask import apply_causal_mask
from .pv_matmul import pv_matmul


def naive_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Attention forward built from four kernels.

    Args:
        q, k, v: [B, H, S, D] fp16 tensors on CUDA.
        causal: whether to apply the upper-triangular causal mask.

    Returns:
        out: [B, H, S, D] fp16.

    Note:
        Softmax is the Chapter 5 row-softmax. We import via torch here
        for chapter readability; replace with the Ch 5 Triton kernel
        when wiring exercises.
    """
    assert q.shape == k.shape == v.shape
    assert q.is_cuda

    scores = qkt_matmul(q, k)            # [B, H, S, S] fp32
    if causal:
        scores = apply_causal_mask(scores)
    probs = F.softmax(scores, dim=-1).to(torch.float16)
    out = pv_matmul(probs, v)            # [B, H, S, D] fp16
    return out
