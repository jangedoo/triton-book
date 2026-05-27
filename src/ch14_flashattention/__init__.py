"""FlashAttention forward (simplified) — Chapter 14.

Online-softmax-based attention forward that never materializes the
[B, H, S, S] score tensor in HBM. Defaults to head_dim=64 and
causal=True; see the chapter exercises for non-causal and other
head_dim variants.
"""

from .flash_attn_fwd import flash_attn_forward, flash_attn_fwd_kernel

__all__ = ["flash_attn_forward", "flash_attn_fwd_kernel"]
