"""Naive (memory-explosive) attention forward built from four Triton kernels.

This package exists to make the cost of materializing the [B, H, S, S]
score matrix visible. Use it as a baseline against the FlashAttention
forward in Chapter 14.
"""

from .naive_attention import naive_attention_forward
from .qkt_matmul import qkt_matmul, qkt_matmul_kernel
from .causal_mask import apply_causal_mask, causal_mask_kernel
from .pv_matmul import pv_matmul, pv_matmul_kernel

__all__ = [
    "naive_attention_forward",
    "qkt_matmul",
    "qkt_matmul_kernel",
    "apply_causal_mask",
    "causal_mask_kernel",
    "pv_matmul",
    "pv_matmul_kernel",
]
