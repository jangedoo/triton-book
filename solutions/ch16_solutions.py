"""Worked solutions for Chapter 16 exercises."""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: Non-interleaved RoPE forward
# See src/ch16_rope/rope.py for the canonical implementation.
# A minimal standalone re-do:
# ---------------------------------------------------------------------------


@triton.jit
def _ex1_rope_noninterleaved(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    sxb, sxh, sxs, sxd,
    sob, soh, sos, sod,
    scs, scd,
    HALF: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)
    b = pid_bh // H; h = pid_bh % H
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)
    s_mask = offs_s < S

    base_x = x_ptr + b * sxb + h * sxh
    x1 = tl.load(base_x + offs_s[:, None] * sxs + offs_h[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base_x + offs_s[:, None] * sxs + (HALF + offs_h)[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + offs_s[:, None] * scs + offs_h[None, :] * scd, mask=s_mask[:, None], other=0.0)
    s = tl.load(sin_ptr + offs_s[:, None] * scs + offs_h[None, :] * scd, mask=s_mask[:, None], other=0.0)

    base_o = out_ptr + b * sob + h * soh
    tl.store(base_o + offs_s[:, None] * sos + offs_h[None, :] * sod, x1 * c - x2 * s, mask=s_mask[:, None])
    tl.store(base_o + offs_s[:, None] * sos + (HALF + offs_h)[None, :] * sod, x1 * s + x2 * c, mask=s_mask[:, None])


# ---------------------------------------------------------------------------
# Exercise 2: Interleaved variant
# See src/ch16_rope/rope.py for the canonical implementation; key change is
# in the dim-index expression for loads/stores.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 3: Cached vs in-kernel cos/sin
# ---------------------------------------------------------------------------


@triton.jit
def _ex3_rope_inline(
    x_ptr, out_ptr,
    B, H, S, D,
    sxb, sxh, sxs, sxd,
    sob, soh, sos, sod,
    BASE_LOG: tl.constexpr,            # log(base), e.g. log(10000)
    HALF: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0); pid_s = tl.program_id(1)
    b = pid_bh // H; h = pid_bh % H
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)
    s_mask = offs_s < S

    # theta[s, i] = s * exp(-2 * i / D * log(base))
    inv_freq = tl.exp(-2.0 * offs_h.to(tl.float32) / D * BASE_LOG)  # [HALF]
    theta = offs_s[:, None].to(tl.float32) * inv_freq[None, :]      # [BLOCK_S, HALF]
    c = tl.cos(theta); s = tl.sin(theta)

    base_x = x_ptr + b * sxb + h * sxh
    x1 = tl.load(base_x + offs_s[:, None] * sxs + offs_h[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base_x + offs_s[:, None] * sxs + (HALF + offs_h)[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)

    base_o = out_ptr + b * sob + h * soh
    tl.store(base_o + offs_s[:, None] * sos + offs_h[None, :] * sod, x1 * c - x2 * s, mask=s_mask[:, None])
    tl.store(base_o + offs_s[:, None] * sos + (HALF + offs_h)[None, :] * sod, x1 * s + x2 * c, mask=s_mask[:, None])


# Benchmark plan:
#   bytes(cache) = S * D/2 * 4 (cos) + S * D/2 * 4 (sin) = S * D * 4 bytes.
#   For S = 4096, D = 128: 2 MiB total. Easily fits in L2.
#
# Expected outcome: at S = 1024, cached and inline are essentially tied
# (both DRAM-bound on x). At S = 4096, inline pulls ahead slightly because
# it saves the cos/sin DRAM reads, which are no longer hot in L2.


# ---------------------------------------------------------------------------
# Exercise 4: Position offset for decoding
# Already implemented in rope_noninterleaved(pos_offset=P).
# See test_rope_pos_offset_decode in tests/test_ch16_rope.py for the
# equivalence check. The trick:
#   pos = offs_s + POS_OFFSET
#   cos_ptrs = cos_ptr + pos[:, None] * stride_cs + ...
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 5: Partial rotary dims
# ---------------------------------------------------------------------------


@triton.jit
def _ex5_rope_partial(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, H, S, D,
    sxb, sxh, sxs, sxd,
    sob, soh, sos, sod,
    scs, scd,
    ROTARY_HALF: tl.constexpr, PASS_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Rotate first 2*ROTARY_HALF dims, copy the remaining PASS_DIM dims through."""
    pid_bh = tl.program_id(0); pid_s = tl.program_id(1)
    b = pid_bh // H; h = pid_bh % H
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, ROTARY_HALF)
    s_mask = offs_s < S

    base_x = x_ptr + b * sxb + h * sxh
    base_o = out_ptr + b * sob + h * soh

    # rotated half
    x1 = tl.load(base_x + offs_s[:, None] * sxs + offs_h[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base_x + offs_s[:, None] * sxs + (ROTARY_HALF + offs_h)[None, :] * sxd, mask=s_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + offs_s[:, None] * scs + offs_h[None, :] * scd, mask=s_mask[:, None], other=0.0)
    s = tl.load(sin_ptr + offs_s[:, None] * scs + offs_h[None, :] * scd, mask=s_mask[:, None], other=0.0)
    tl.store(base_o + offs_s[:, None] * sos + offs_h[None, :] * sod, x1 * c - x2 * s, mask=s_mask[:, None])
    tl.store(base_o + offs_s[:, None] * sos + (ROTARY_HALF + offs_h)[None, :] * sod, x1 * s + x2 * c, mask=s_mask[:, None])

    # pass-through tail: copy [2*ROTARY_HALF, 2*ROTARY_HALF + PASS_DIM)
    if PASS_DIM > 0:
        offs_p = tl.arange(0, PASS_DIM)
        passthru = tl.load(
            base_x + offs_s[:, None] * sxs + (2 * ROTARY_HALF + offs_p)[None, :] * sxd,
            mask=s_mask[:, None], other=0.0,
        )
        tl.store(
            base_o + offs_s[:, None] * sos + (2 * ROTARY_HALF + offs_p)[None, :] * sod,
            passthru, mask=s_mask[:, None],
        )


# ---------------------------------------------------------------------------
# Exercise 6: Fuse RoPE into Q/K projection (sketch)
# ---------------------------------------------------------------------------


# Full implementation is out of scope (depends on Chapter 10's matmul
# infrastructure). The pattern:
#
#   1. The matmul kernel produces an accumulator tile `acc` of shape
#      [BLOCK_M, BLOCK_D] holding a chunk of Q (one head's worth, all D dims).
#   2. After the K-loop, immediately before storing `acc`, split it into the
#      first-half / second-half halves on the D axis.
#   3. Apply RoPE in-register:
#        rot1 = h1 * cos - h2 * sin
#        rot2 = h1 * sin + h2 * cos
#      where cos, sin are loaded once at the top of the kernel using the
#      program's M-offset as the position.
#   4. Store rot1, rot2 to the corresponding halves of the output tile.
#
# You save one full read + one full write of Q per token per layer.
