"""Worked solutions for Chapter 29 exercises."""

from __future__ import annotations

import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "ch28_mini_lib"))

from ch29_transformer_swap.pytorch_block import (  # noqa: E402
    BlockConfig,
    PyTorchBlock,
    _apply_rope_torch,
    _rmsnorm_torch,
    _rope_tables,
)
from mini_triton_llm import flash_attention, rmsnorm, rope, swiglu  # noqa: E402
from mini_triton_llm.benchmarking import bench  # noqa: E402


# ---------------------------------------------------------------------------
# Exercise 1: swap only RMSNorm.
# ---------------------------------------------------------------------------
class OnlyRMSNormSwapped(PyTorchBlock):
    def forward(self, x):
        B, S, _ = x.shape
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        self._maybe_build_rope(x.device)

        h = rmsnorm(x, self.norm1_weight, eps=cfg.rms_eps)  # <-- swapped
        q = self.q_proj(h).view(B, S, H, D)
        k = self.k_proj(h).view(B, S, H, D)
        v = self.v_proj(h).view(B, S, H, D)
        q = _apply_rope_torch(q, self._cos, self._sin)
        k = _apply_rope_torch(k, self._cos, self._sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)

        h2 = rmsnorm(x, self.norm2_weight, eps=cfg.rms_eps)  # <-- swapped
        gated = self.up_proj(h2)
        a, b = gated.chunk(2, dim=-1)
        x = x + self.down_proj(F.silu(a) * b)
        return x


# ---------------------------------------------------------------------------
# Exercise 2: swap only SwiGLU.
# ---------------------------------------------------------------------------
class OnlySwigluSwapped(PyTorchBlock):
    def forward(self, x):
        B, S, _ = x.shape
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        self._maybe_build_rope(x.device)

        h = _rmsnorm_torch(x, self.norm1_weight, cfg.rms_eps)
        q = self.q_proj(h).view(B, S, H, D)
        k = self.k_proj(h).view(B, S, H, D)
        v = self.v_proj(h).view(B, S, H, D)
        q = _apply_rope_torch(q, self._cos, self._sin)
        k = _apply_rope_torch(k, self._cos, self._sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)

        h2 = _rmsnorm_torch(x, self.norm2_weight, cfg.rms_eps)
        gated = self.up_proj(h2)
        a, b = gated.chunk(2, dim=-1)
        x = x + self.down_proj(swiglu(a, b))  # <-- swapped
        return x


# ---------------------------------------------------------------------------
# Exercise 3: swap only attention.
# ---------------------------------------------------------------------------
class OnlyAttnSwapped(PyTorchBlock):
    def forward(self, x):
        B, S, _ = x.shape
        cfg = self.cfg
        H, D = cfg.num_heads, cfg.head_dim
        self._maybe_build_rope(x.device)

        h = _rmsnorm_torch(x, self.norm1_weight, cfg.rms_eps)
        q = self.q_proj(h).view(B, S, H, D)
        k = self.k_proj(h).view(B, S, H, D)
        v = self.v_proj(h).view(B, S, H, D)
        q = _apply_rope_torch(q, self._cos, self._sin)
        k = _apply_rope_torch(k, self._cos, self._sin)
        q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        attn = flash_attention(q, k, v, causal=True)  # <-- swapped
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        x = x + self.o_proj(attn)

        h2 = _rmsnorm_torch(x, self.norm2_weight, cfg.rms_eps)
        gated = self.up_proj(h2)
        a, b = gated.chunk(2, dim=-1)
        x = x + self.down_proj(F.silu(a) * b)
        return x


# ---------------------------------------------------------------------------
# Exercise 4: a 4-layer stack.
# ---------------------------------------------------------------------------
def build_stack(block_cls, cfg: BlockConfig, n_layers: int = 4, dtype=torch.float16):
    return nn.Sequential(*[block_cls(cfg, dtype=dtype) for _ in range(n_layers)])


def bench_stack():
    cfg = BlockConfig()
    torch_stack = build_stack(PyTorchBlock, cfg).cuda().eval()
    # Build TritonBlocks with the same weights, layer by layer.
    from ch29_transformer_swap import TritonBlock

    triton_stack = nn.Sequential(
        *[TritonBlock.from_pytorch(b) for b in torch_stack]
    ).cuda().eval()

    B, S = 4, 1024
    x = torch.randn(B, S, cfg.hidden_dim, dtype=torch.float16, device="cuda") * 0.1

    with torch.no_grad():
        t_torch = bench(lambda: torch_stack(x))
        t_tri = bench(lambda: triton_stack(x))
    print(f"4-layer torch: {t_torch:.3f} ms   triton: {t_tri:.3f} ms")


# ---------------------------------------------------------------------------
# Exercise 5: KV-cache decode loop sketch.
# ---------------------------------------------------------------------------
def decode_step(block, x_token, k_cache, v_cache, past_len):
    """Run one new-token forward through a TritonBlock-flavored module.

    The module needs to expose `q/k/v/o/up/down` projections, the
    `norm1/norm2` weights, and the RoPE tables. We reuse the existing
    TritonBlock and patch the attention path for cached K, V.
    """
    cfg = block.cfg
    H, D = cfg.num_heads, cfg.head_dim
    block._maybe_build_rope(x_token.device)

    h = rmsnorm(x_token, block.norm1_weight, eps=cfg.rms_eps)
    q = block.q_proj(h).view(1, 1, H, D)
    k_new = block.k_proj(h).view(1, 1, H, D)
    v_new = block.v_proj(h).view(1, 1, H, D)

    cos = block._cos
    sin = block._sin
    # Apply RoPE at position `past_len`.
    q = rope(q, cos, sin, offset=past_len)
    k_new = rope(k_new, cos, sin, offset=past_len)

    # Append to cache.
    if k_cache is None:
        k_full = k_new
        v_full = v_new
    else:
        k_full = torch.cat([k_cache, k_new], dim=1)
        v_full = torch.cat([v_cache, v_new], dim=1)

    q = q.transpose(1, 2).contiguous()
    k_full_h = k_full.transpose(1, 2).contiguous()
    v_full_h = v_full.transpose(1, 2).contiguous()
    attn = flash_attention(q, k_full_h, v_full_h, causal=True)
    attn = attn.transpose(1, 2).contiguous().view(1, 1, -1)
    out = x_token + block.o_proj(attn)
    h2 = rmsnorm(out, block.norm2_weight, eps=cfg.rms_eps)
    gated = block.up_proj(h2)
    a, b = gated.chunk(2, dim=-1)
    out = out + block.down_proj(swiglu(a, b))
    return out, k_full, v_full


# ---------------------------------------------------------------------------
# Exercise 6: Triton backward via PyTorch fallback (the "honest" option).
# Use the PyTorchBlock as a drop-in for the backward; you still get the
# Triton forward speed. The proper solution wraps each Triton kernel in
# torch.autograd.Function with its own backward kernel; see Chapter 15
# for the attention backward.
# ---------------------------------------------------------------------------
def autograd_fallback_demo():
    cfg = BlockConfig(hidden_dim=64, num_heads=2, head_dim=32, intermediate_dim=256)
    ref = PyTorchBlock(cfg, dtype=torch.float32).cuda()
    x = torch.randn(2, 16, cfg.hidden_dim, dtype=torch.float32, device="cuda", requires_grad=True)
    y = ref(x).sum()
    y.backward()
    print("PyTorch backward works as-is; Triton backward requires wrapping.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        bench_stack()
        autograd_fallback_demo()
