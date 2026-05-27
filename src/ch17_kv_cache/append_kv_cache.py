"""Append a single token's K and V into a pre-allocated KV cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _append_kv_kernel(
    k_new_ptr, v_new_ptr,
    k_cache_ptr, v_cache_ptr,
    B, H, D, POSITION,
    stride_knb, stride_knh, stride_kns, stride_knd,
    stride_vnb, stride_vnh, stride_vns, stride_vnd,
    stride_kcb, stride_kch, stride_kcs, stride_kcd,
    stride_vcb, stride_vch, stride_vcs, stride_vcd,
    BLOCK_D: tl.constexpr,
):
    """Write k_new[b, h, 0, :] -> k_cache[b, h, POSITION, :] and likewise for V."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh %  H

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    k_new_ptrs = k_new_ptr + b * stride_knb + h * stride_knh + offs_d * stride_knd
    v_new_ptrs = v_new_ptr + b * stride_vnb + h * stride_vnh + offs_d * stride_vnd

    k = tl.load(k_new_ptrs, mask=d_mask, other=0.0)
    v = tl.load(v_new_ptrs, mask=d_mask, other=0.0)

    k_cache_ptrs = (
        k_cache_ptr + b * stride_kcb + h * stride_kch
        + POSITION * stride_kcs + offs_d * stride_kcd
    )
    v_cache_ptrs = (
        v_cache_ptr + b * stride_vcb + h * stride_vch
        + POSITION * stride_vcs + offs_d * stride_vcd
    )
    tl.store(k_cache_ptrs, k, mask=d_mask)
    tl.store(v_cache_ptrs, v, mask=d_mask)


def append_kv_cache(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: int,
) -> None:
    """Append k_new, v_new (shape [B, H, 1, D]) into the cache at `position`.

    The cache tensors are mutated in place. No return value.
    """
    assert k_new.shape == v_new.shape, "k_new and v_new must match"
    assert k_new.shape[2] == 1, "expected a single token: shape [B, H, 1, D]"
    assert k_new.is_cuda
    B, H, _, D = k_new.shape
    max_seq = k_cache.shape[2]
    assert 0 <= position < max_seq, f"position {position} out of bounds [0, {max_seq})"
    assert k_cache.shape == v_cache.shape == (B, H, max_seq, D)

    BLOCK_D = max(16, triton.next_power_of_2(D))
    grid = (B * H,)
    _append_kv_kernel[grid](
        k_new, v_new, k_cache, v_cache,
        B, H, D, position,
        *k_new.stride(), *v_new.stride(),
        *k_cache.stride(), *v_cache.stride(),
        BLOCK_D=BLOCK_D,
    )
