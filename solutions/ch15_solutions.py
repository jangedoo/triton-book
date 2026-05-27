"""Worked solutions for Chapter 15 exercises."""

# Exercise 1: Re-derive the softmax-row Jacobian.
#
# p_i = exp(s_i) / Z, where Z = sum_j exp(s_j).
#
# Diagonal case (i == j):
#   dp_i / ds_i = exp(s_i) / Z - exp(s_i) * exp(s_i) / Z^2
#               = p_i - p_i * p_i
#               = p_i * (1 - p_i)
#
# Off-diagonal (i != j):
#   dp_i / ds_j = - exp(s_i) * exp(s_j) / Z^2
#               = - p_i * p_j
#
# Unify: dp_i / ds_j = p_i * (delta_ij - p_j).
#
# Pull dP through the Jacobian:
#   dS_i = sum_j dP_j * (dp_j / ds_i)
#        = sum_j dP_j * p_j * (delta_ji - p_i)
#        = dP_i * p_i  -  p_i * sum_j dP_j * p_j
#        = p_i * (dP_i - rowsum(dP_i * P_i))
#
# Vectorized over rows: dS = P * (dP - rowsum(dP * P, keepdim=True)).


# Exercise 2: Implement just dV as a kernel.
#
# See _dv_kernel in src/ch15_attention_backward/educational_backward.py.
# A minimal standalone launcher:

import math
import torch
import triton

from src.ch15_attention_backward.educational_backward import _dv_kernel


def dv_only(p: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    B, H, S, _ = p.shape
    D = do.shape[-1]
    dv = torch.empty(B, H, S, D, device=p.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(16, triton.next_power_of_2(D))
    p32  = p.to(torch.float32)
    do32 = do.to(torch.float32)
    grid = (B * H, triton.cdiv(S, BLOCK_N))
    _dv_kernel[grid](
        p32, do32, dv,
        B, H, S, D,
        *p32.stride(), *do32.stride(), *dv.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return dv


# Exercise 3: FLOPs vs bytes for recompute.
#
# For B=1, H=16, S=4096, D=64:
#
# Storing P (fp32):     1 * 16 * 4096 * 4096 * 4 bytes
#                     = 1,073,741,824 bytes
#                     = 1.0 GiB per (B, H, S, S) tensor.
#
# Recomputing P once:
#   Q @ K^T:           2 * 1 * 16 * 4096 * 4096 * 64 FLOPs
#                    = 34,359,738,368 FLOPs (about 34 GFLOPs).
#   softmax row:       ~5 * S^2 * B * H FLOPs (negligible vs the matmul).
#
# An A100 sustains ~150-300 TFLOP/s in fp16. 34 GFLOPs is ~0.1-0.2 ms of
# compute. Loading 1 GiB at 1.5 TB/s (HBM peak) is ~0.7 ms, and you'd pay it
# every time the backward kernel touched P. Recompute wins handily even
# before you consider that 1 GiB might not fit alongside Q, K, V, dO.


# Exercise 4: dQ recomputation skeleton.
#
# See _dq_kernel in src/ch15_attention_backward/educational_backward.py.
# Standalone launcher:

from src.ch15_attention_backward.educational_backward import _dq_kernel


def dq_only(ds: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    B, H, S, _ = ds.shape
    D = k.shape[-1]
    dq = torch.empty(B, H, S, D, device=ds.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = max(16, triton.next_power_of_2(D))
    ds32 = ds.to(torch.float32)
    k32  = k.to(torch.float32)
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _dq_kernel[grid](
        ds32, k32, dq,
        B, H, S, D, scale,
        *ds32.stride(), *k32.stride(), *dq.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return dq


# Exercise 5: _attn_bwd_preprocess explanation.
#
# _attn_bwd_preprocess computes delta = rowsum(O * dO) per row, shape
# [B, H, S], in fp32. The identity rowsum(dP * P) = rowsum(dO * O) lets the
# main kernels (_attn_bwd_dq and _attn_bwd_dkdv) skip a fresh reduction over
# the [S, S] dP * P matrix every time they need the softmax-row correction
# term. Both kernels need delta (it appears in dS = P * (dP - delta)) and
# would otherwise either materialize dP or recompute the reduction. The
# right-hand side, dO * O, is element-wise over [B, H, S, D] tensors that
# the launcher already has on hand and can reduce once.


# Exercise 6: Causal backward.
#
# Add `causal=True` and apply the mask in two places:
#  - in the forward recompute of s/p, set s[m, n] = -inf for n > m before
#    softmax.
#  - when computing dS, those positions already have p = 0 so dS is 0 there
#    automatically; no extra mask needed.
#
# Naive approach (slow but obviously correct):

def attention_backward_educational_causal(q, k, v, do):
    import math
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    q32, k32, v32, do32 = (t.to(torch.float32) for t in (q, k, v, do))
    s = (q32 @ k32.transpose(-1, -2)) * scale
    causal_mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
    )
    s = s.masked_fill(causal_mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    dp = do32 @ v32.transpose(-1, -2)
    row = (dp * p).sum(dim=-1, keepdim=True)
    ds = p * (dp - row)
    dv = p.transpose(-1, -2) @ do32
    dq = (ds @ k32) * scale
    dk = (ds.transpose(-1, -2) @ q32) * scale
    return dq, dk, dv

# Triton version: pass a causal flag into _dq_kernel and _dk_kernel; inside
# the inner loop, skip tiles where the entire tile is masked
# (m_tile_max < n_tile_min for dQ; symmetric for dK/dV). For boundary tiles,
# apply a per-element mask before the tl.dot accumulation.
