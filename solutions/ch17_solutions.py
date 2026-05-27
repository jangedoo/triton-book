"""Worked solutions for Chapter 17 exercises."""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: append kernel — see src/ch17_kv_cache/append_kv_cache.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 2: contiguous-gather kernel
# ---------------------------------------------------------------------------


@triton.jit
def _gather_kernel(
    src_ptr, dst_ptr,
    B, H, MAX_SEQ, CURRENT_LEN, D,
    stride_sb, stride_sh, stride_ss, stride_sd,
    stride_db, stride_dh, stride_ds, stride_dd,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)
    b = pid_bh // H; h = pid_bh % H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    s_mask = offs_s < CURRENT_LEN
    d_mask = offs_d < D

    src_ptrs = (
        src_ptr + b * stride_sb + h * stride_sh
        + offs_s[:, None] * stride_ss + offs_d[None, :] * stride_sd
    )
    dst_ptrs = (
        dst_ptr + b * stride_db + h * stride_dh
        + offs_s[:, None] * stride_ds + offs_d[None, :] * stride_dd
    )
    data = tl.load(src_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
    tl.store(dst_ptrs, data, mask=s_mask[:, None] & d_mask[None, :])


def gather_cache(k_cache: torch.Tensor, current_len: int) -> torch.Tensor:
    B, H, _, D = k_cache.shape
    out = torch.empty(B, H, current_len, D, device=k_cache.device, dtype=k_cache.dtype)
    BLOCK_S = 32
    BLOCK_D = max(16, triton.next_power_of_2(D))
    grid = (B * H, triton.cdiv(current_len, BLOCK_S))
    _gather_kernel[grid](
        k_cache, out,
        B, H, k_cache.shape[2], current_len, D,
        *k_cache.stride(), *out.stride(),
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 3: append throughput by batch size
# Run benchmarks/bench_ch17_kv_cache.py main(). Expected pattern:
#   B = 1:  launch-overhead bound; ms is flat against bytes
#   B = 4:  intermediate
#   B = 16: bandwidth-bound; ms scales ~linearly with B
#   B = 64: cleanly bandwidth-bound
# Cross-over point: roughly the B where bytes / DRAM_BW ~ 5us. For
# H=32, D=128, fp16:
#   bytes per launch = 2 (K,V) * 2 (read+write) * B * 32 * 128 * 2 bytes
#                    = B * 32768 bytes
# DRAM at 1.5 TB/s: 5 us = 7.5 MB, so cross-over around B = 230. Below
# that, launch overhead dominates. (Numbers will differ on your GPU.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 4: GQA decode
# ---------------------------------------------------------------------------


@triton.jit
def _gqa_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    B, H_Q, H_KV, CURRENT_LEN, D, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per (b, h_kv). Iterates the GROUP_SIZE query heads
    that share this KV head, reading the cache once per group."""
    pid_bh = tl.program_id(0)
    b    = pid_bh // H_KV
    h_kv = pid_bh %  H_KV

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    # Load all GROUP_SIZE query heads for this (b, h_kv) into a [GROUP_SIZE, BLOCK_D] tile.
    offs_g = tl.arange(0, GROUP_SIZE)
    h_q = h_kv * GROUP_SIZE + offs_g                # [GROUP_SIZE]
    q_ptrs = (
        q_ptr + b * stride_qb
        + h_q[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32) * scale

    m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
    acc = tl.zeros((GROUP_SIZE, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, CURRENT_LEN, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < CURRENT_LEN
        k_ptrs = (
            k_cache_ptr + b * stride_kb + h_kv * stride_kh
            + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_cache_ptr + b * stride_vb + h_kv * stride_vh
            + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Scores per query head: [GROUP_SIZE, BLOCK_N] = q [G, D] @ k^T [D, BLOCK_N]
        s = tl.dot(q, tl.trans(k_tile))
        s = tl.where(n_mask[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p, v_tile)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptrs = (
        out_ptr + b * stride_ob
        + h_q[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=d_mask[None, :])


def gqa_decode_attention(q, k_cache, v_cache, current_len):
    """q: [B, H_q, 1, D]; k_cache,v_cache: [B, H_kv, max_seq, D]."""
    B, H_q, _, D = q.shape
    H_kv = k_cache.shape[1]
    assert H_q % H_kv == 0
    GROUP_SIZE = H_q // H_kv
    scale = 1.0 / math.sqrt(D)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    BLOCK_N = 64
    out = torch.empty_like(q)
    grid = (B * H_kv,)
    _gqa_decode_kernel[grid](
        q, k_cache, v_cache, out,
        B, H_q, H_kv, current_len, D, scale,
        *q.stride(), *k_cache.stride(), *v_cache.stride(), *out.stride(),
        GROUP_SIZE=GROUP_SIZE, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 5: block-table lookup primitive
# ---------------------------------------------------------------------------


@triton.jit
def _block_lookup_kernel(
    block_table_ptr, out_ptr,
    B, MAX_BLOCKS, NUM_POS,
    stride_tb, stride_tm,
    stride_ob, stride_op,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_POS: tl.constexpr,
):
    """phys[b, p] = block_table[b, p // BLOCK_TOKENS] * BLOCK_TOKENS + p % BLOCK_TOKENS."""
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_p = pid_p * BLOCK_POS + tl.arange(0, BLOCK_POS)
    p_mask = offs_p < NUM_POS

    block_idx = offs_p // BLOCK_TOKENS
    slot      = offs_p %  BLOCK_TOKENS

    bt_ptrs = block_table_ptr + pid_b * stride_tb + block_idx * stride_tm
    blk = tl.load(bt_ptrs, mask=p_mask, other=0)
    phys = blk * BLOCK_TOKENS + slot

    out_ptrs = out_ptr + pid_b * stride_ob + offs_p * stride_op
    tl.store(out_ptrs, phys, mask=p_mask)


def block_lookup(block_table: torch.Tensor, num_pos: int, block_tokens: int) -> torch.Tensor:
    B, MAX_BLOCKS = block_table.shape
    out = torch.empty(B, num_pos, device=block_table.device, dtype=torch.int32)
    BLOCK_POS = 64
    grid = (B, triton.cdiv(num_pos, BLOCK_POS))
    _block_lookup_kernel[grid](
        block_table.to(torch.int32), out,
        B, MAX_BLOCKS, num_pos,
        *block_table.stride(), *out.stride(),
        BLOCK_TOKENS=block_tokens, BLOCK_POS=BLOCK_POS,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 6: tiny paged-decode attention (sketch)
# ---------------------------------------------------------------------------


# A full implementation has many moving parts. The structural skeleton:
#
# @triton.jit
# def _paged_decode_kernel(
#     q_ptr, k_pool_ptr, v_pool_ptr, out_ptr,
#     block_table_ptr, current_len_ptr,
#     H, D, scale,
#     ... strides ...
#     BLOCK_TOKENS: tl.constexpr, BLOCK_D: tl.constexpr,
# ):
#     pid_bh = tl.program_id(0)
#     b = pid_bh // H; h = pid_bh % H
#     current_len = tl.load(current_len_ptr + b)
#
#     offs_d = tl.arange(0, BLOCK_D); d_mask = offs_d < D
#     q = tl.load(q_ptr + ...).to(tl.float32) * scale
#
#     m_i = -inf; l_i = 0; acc = zeros(BLOCK_D)
#     num_blocks = cdiv(current_len, BLOCK_TOKENS)
#     for block_i in range(num_blocks):
#         phys = tl.load(block_table_ptr + b * stride_tb + block_i * stride_tm)
#         offs_n_in_block = tl.arange(0, BLOCK_TOKENS)
#         valid = block_i * BLOCK_TOKENS + offs_n_in_block < current_len
#         k_ptrs = k_pool_ptr + phys * stride_block + offs_n_in_block[:, None] * stride_tok + h * stride_h + offs_d[None, :] * stride_d
#         v_ptrs = v_pool_ptr + phys * stride_block + offs_n_in_block[:, None] * stride_tok + h * stride_h + offs_d[None, :] * stride_d
#         k_tile = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
#         v_tile = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
#         # ... online softmax as before, masking invalid lanes to -inf
#
# Verification recipe:
#   1. Build a contiguous cache from the paged pool with a host-side gather.
#   2. Run src/ch17_kv_cache/decode_attention.py on the contiguous cache.
#   3. Run the paged kernel.
#   4. assert_close.
