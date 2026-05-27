"""Worked solutions for Chapter 11 exercises.

Most are prose because the underlying kernels target hardware the author
cannot validate against. Where code is provided, it is the smallest
possible illustrative version.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Exercise 1: persistent + grouped, why they combine
# ---------------------------------------------------------------------------
# Persistent execution keeps the *same physical SM* working on tiles
# back-to-back. Grouped ordering ensures that the tiles a single
# program/SM touches share rows of A across many columns of B. Taken
# together: the same SM walks a row-stripe of A through L1/L2, all the
# while consuming column-stripes of B that get cached in L2 too. You
# extract reuse from both A and B without needing to materialize bigger
# tiles or spill registers.


# ---------------------------------------------------------------------------
# Exercise 2: the official tutorial vs our skeleton
# ---------------------------------------------------------------------------
# The official Triton persistent matmul tutorial (09-persistent-matmul.py)
# does several things our skeleton does not:
#   - Software-pipelines the K-loop with explicit `tl.dot` accumulator
#     chaining so the next load is in flight while the current dot runs.
#   - Uses `num_stages` and `num_warps` autotuned per-config.
#   - On Hopper, swaps the loads for TMA (Tensor Memory Accelerator)
#     descriptors so the address arithmetic stops being per-thread.
#   - Handles the epilogue (cast, store) as a fused pipeline stage so
#     the store does not stall the next K-loop start.
# The outer `for tile_id in range(start_pid, num_tiles, NUM_SMS):`
# structure is the same.


# ---------------------------------------------------------------------------
# Exercise 3: launch-overhead estimate
# ---------------------------------------------------------------------------
# Regular kernel: 1024 programs * 5 us = 5120 us = 5.12 ms total launch
# overhead. If the kernel takes 2 ms, the launch overhead alone (5.12 ms)
# exceeds the kernel time -- which means the model is wrong, or the
# overhead is amortized across SMs running in parallel. The right model:
# launch overhead is *per kernel launch*, not per program. Programs are
# scheduled to SMs once the kernel is launched.
#
# A better question: what is the total *scheduling* overhead?
# Per-program scheduling on modern hardware is roughly 10-100 ns. So:
#   - Regular: 1024 * 50 ns = ~50 us per kernel launch.
#   - Persistent: 84 * 50 ns = ~4 us per kernel launch.
# For a 2 ms kernel, that is 2.5% vs 0.2% -- noticeable on smaller
# shapes, marginal at scale. The bigger win for persistent is the
# pipelining and TMA overlap, not the raw launch saving.


# ---------------------------------------------------------------------------
# Exercise 4: persistent variant of Chapter 9 grouped matmul
# ---------------------------------------------------------------------------
# See src/ch11_persistent_matmul/persistent_matmul_skeleton.py for a
# complete implementation. The only structural change from the Chapter 9
# grouped kernel is wrapping the tile body in:
#
#   start_pid = tl.program_id(axis=0)
#   for tile_id in range(start_pid, num_tiles, NUM_SMS):
#       ... existing grouped-mapping + K-loop + epilogue ...
#
# and launching with `grid = (NUM_SMS,)` instead of
# `(num_pid_m * num_pid_n,)`. Correctness should match bit-for-bit
# (same dot order, same fp32 accumulator) regardless of GPU.


# ---------------------------------------------------------------------------
# Exercise 5: FP8 matmul kernel signature sketch
# ---------------------------------------------------------------------------
# Compared to fp16, an FP8 matmul kernel typically adds:
#   - a_scale_ptr, b_scale_ptr  (per-block fp16 or fp32 scales)
#   - stride args for the scale tensors
#   - a SCALE_BLOCK constexpr telling the kernel how many fp8 elements
#     share one scale (commonly 32 or 128 along the K axis).
#
# Pseudo-signature:
#
#   @triton.jit
#   def fp8_matmul_kernel(
#       a_ptr, b_ptr,            # fp8 e4m3 or e5m2
#       a_scale_ptr, b_scale_ptr, # fp16 or fp32
#       c_ptr,                   # bf16 or fp16 output
#       M, N, K,
#       stride_am, stride_ak,
#       stride_bk, stride_bn,
#       stride_as_m, stride_as_k, # scale strides
#       stride_bs_k, stride_bs_n,
#       stride_cm, stride_cn,
#       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
#       BLOCK_K: tl.constexpr,
#       SCALE_BLOCK: tl.constexpr,
#   ):
#       ...
#       a = tl.load(a_ptrs)            # fp8
#       b = tl.load(b_ptrs)            # fp8
#       a_s = tl.load(a_scale_ptrs)    # fp16 or fp32
#       b_s = tl.load(b_scale_ptrs)    # fp16 or fp32
#       # Convert+rescale to fp32 just before the dot:
#       partial = tl.dot(a, b)         # fp32
#       acc += partial * a_s * b_s     # apply per-block scale
#
# The per-block scaling lets activations with occasional outliers stay
# representable: each block has its own range, instead of one tensor-wide
# range that has to cover the outlier.


# ---------------------------------------------------------------------------
# Exercise 6: block-scaled matmul tutorial summary (prose)
# ---------------------------------------------------------------------------
# (a) Scaling-tensor layout: the official block-scaled tutorial stores
#     scales in a separate small tensor whose shape matches the matmul
#     operand's tile partitioning. For a (M, K) fp8 matrix with
#     SCALE_BLOCK along K, the scale tensor is (M, K / SCALE_BLOCK)
#     fp16 (or microexponent fp8 on Blackwell). Each scale entry covers
#     one row-stripe of one BLOCK_K-sized window of the operand.
#
# (b) The kernel walks both the data pointer and the scale pointer in
#     lockstep through the K-loop. On each iteration it loads the
#     BLOCK_K x BLOCK_N data tile and the corresponding (1 x BLOCK_N)
#     or (BLOCK_M x 1) scale slice, dots the data, then multiplies the
#     fp32 partial product by the scales before adding to the
#     accumulator. The scale loads are tiny relative to the data and
#     fit in registers.
#
# (c) Per-block scaling beats per-tensor scaling for outliers: one
#     extreme activation in one column does not force every other
#     element in the tensor to share its huge dynamic range. The block
#     containing the outlier gets its own large scale; the rest keep
#     their precision. Blackwell's microscaling (MX-FP4 / MX-FP8) takes
#     this further -- 32-element blocks with their own exponents.
