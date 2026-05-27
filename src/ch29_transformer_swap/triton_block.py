"""Triton transformer block.

Same shape contract as `PyTorchBlock`; same parameter set; same forward
math. Only the four hot ops are swapped:

    PyTorch RMSNorm                    -> mini_triton_llm.rmsnorm
    PyTorch RoPE                       -> mini_triton_llm.rope
    F.scaled_dot_product_attention     -> mini_triton_llm.flash_attention
    F.silu(a) * b                      -> mini_triton_llm.swiglu

Linears stay on cuBLAS. There is no point reimplementing matmul here;
Chapter 9 covered that, and cuBLAS will win.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Wire `mini_triton_llm` into sys.path so this module is usable without
# installing the package.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "ch28_mini_lib"))

from mini_triton_llm import flash_attention, rmsnorm, rope, swiglu  # noqa: E402

from .pytorch_block import BlockConfig, PyTorchBlock, _rope_tables


class TritonBlock(nn.Module):
    """Same forward as `PyTorchBlock`, four kernels swapped."""

    def __init__(self, cfg: BlockConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.num_heads, cfg.head_dim
        hidden = cfg.hidden_dim
        assert hidden == H * D

        self.norm1_weight = nn.Parameter(torch.ones(hidden, dtype=dtype))
        self.norm2_weight = nn.Parameter(torch.ones(hidden, dtype=dtype))

        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)

        self.up_proj = nn.Linear(hidden, cfg.intermediate_dim * 2, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(cfg.intermediate_dim, hidden, bias=False, dtype=dtype)

        self._cos = None
        self._sin = None
        self._dtype = dtype

    @classmethod
    def from_pytorch(cls, ref: PyTorchBlock) -> "TritonBlock":
        """Build a TritonBlock sharing weights with a PyTorchBlock."""
        out = cls(ref.cfg, dtype=ref._dtype)
        out.load_state_dict(ref.state_dict(), strict=True)
        return out

    def _maybe_build_rope(self, device: torch.device):
        if self._cos is None or self._cos.device != device:
            self._cos, self._sin = _rope_tables(self.cfg, device, self._dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        self._maybe_build_rope(x.device)

        # ---- attention ----
        h = rmsnorm(x, self.norm1_weight, eps=cfg.rms_eps)
        q = self.q_proj(h).view(B, S, H, D)
        k = self.k_proj(h).view(B, S, H, D)
        v = self.v_proj(h).view(B, S, H, D)

        q = rope(q, self._cos, self._sin, offset=0)
        k = rope(k, self._cos, self._sin, offset=0)

        # flash_attention expects (B, H, S, D)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        attn = flash_attention(q, k, v, causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)

        # ---- MLP ----
        h2 = rmsnorm(x, self.norm2_weight, eps=cfg.rms_eps)
        gated = self.up_proj(h2)
        a, b = gated.chunk(2, dim=-1)
        mlp = swiglu(a, b)
        x = x + self.down_proj(mlp)
        return x
