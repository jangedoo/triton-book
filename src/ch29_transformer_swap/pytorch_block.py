"""PyTorch reference transformer block.

Pre-norm, RMSNorm, RoPE on (Q, K), causal multi-head self-attention via
`F.scaled_dot_product_attention`, residual, RMSNorm, SwiGLU MLP, residual.

The block is intentionally a one-file blob: linear layers, the RoPE
tables, the helpers, all in plain PyTorch. The Triton version in
`triton_block.py` shares the *same* sub-module weights, so we can swap
just the math kernels and confirm correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BlockConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    head_dim: int = 64
    intermediate_dim: int = 1024
    rms_eps: float = 1e-6
    rope_base: float = 10000.0
    max_seq_len: int = 4096


def _rope_tables(cfg: BlockConfig, device: torch.device, dtype: torch.dtype):
    D = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
    pos = torch.arange(cfg.max_seq_len, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, D)
    D = x.shape[-1]
    x0, x1 = x[..., : D // 2], x[..., D // 2 :]
    S = x.shape[1]
    c = cos[None, :S, None, :]
    s = sin[None, :S, None, :]
    return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)


def _rmsnorm_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = (x.float() ** 2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * w.float()).to(x.dtype)


class PyTorchBlock(nn.Module):
    """Reference block. Forward is intentionally plain so it acts as oracle."""

    def __init__(self, cfg: BlockConfig, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.num_heads, cfg.head_dim
        hidden = cfg.hidden_dim
        assert hidden == H * D, "hidden_dim must equal num_heads * head_dim"

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

    def _maybe_build_rope(self, device: torch.device):
        if self._cos is None or self._cos.device != device:
            self._cos, self._sin = _rope_tables(self.cfg, device, self._dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        self._maybe_build_rope(x.device)

        # ---- attention sub-block ----
        h = _rmsnorm_torch(x, self.norm1_weight, cfg.rms_eps)
        q = self.q_proj(h).view(B, S, H, D)
        k = self.k_proj(h).view(B, S, H, D)
        v = self.v_proj(h).view(B, S, H, D)

        q = _apply_rope_torch(q, self._cos, self._sin)
        k = _apply_rope_torch(k, self._cos, self._sin)

        # (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)

        # ---- MLP sub-block ----
        h2 = _rmsnorm_torch(x, self.norm2_weight, cfg.rms_eps)
        gated = self.up_proj(h2)  # (B, S, 2I)
        a, b = gated.chunk(2, dim=-1)
        mlp = F.silu(a) * b
        x = x + self.down_proj(mlp)
        return x
