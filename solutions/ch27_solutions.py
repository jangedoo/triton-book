"""Solutions for Chapter 27 — diagnosis puzzles.

Numbers are for the 2070 SUPER (peak DRAM ~448 GB/s, peak fp16 tensor-core
~57 TFLOP/s). Adjust to your hardware.
"""


# ---------------------------------------------------------------------------
# Puzzle 1: layernorm on (1024, 4096) fp16.
# Bytes:
#     read input:     1024 * 4096 * 2 = 8,388,608
#     write output:   1024 * 4096 * 2 = 8,388,608
#     read w + b:     4096 * 2 * 2    = 16,384  (small)
#     total ≈ 16,793,600 bytes
# Time: 0.40 ms = 4e-4 s
# GB/s = 1.679e7 / 4e-4 / 1e9 = 41.9 GB/s
# ---------------------------------------------------------------------------
# Diagnosis: 42 GB/s is ~10% of peak DRAM. The kernel is memory-bound (this
# is layernorm — no significant compute), and it is leaving most of the
# bandwidth on the floor.
#
# Next experiment: profile to find the bottleneck.
#   1) Is the kernel making multiple passes over the input (compute mean, then
#      compute variance, then normalize)? If so, fuse to a single-pass Welford
#      or two-pass-in-shared-memory layout.
#   2) Is BLOCK_SIZE matched to N (=4096)? An undersized BLOCK forces the kernel
#      to walk the row in multiple iterations and incurs extra control overhead.
#   3) Add autotune over (BLOCK_SIZE, num_warps).
def puzzle_1():
    bytes_total = 1024 * 4096 * 2 * 2 + 4096 * 2 * 2
    ms = 0.40
    gbs = bytes_total / (ms * 1e-3) / 1e9
    print(f"puzzle 1: {gbs:.1f} GB/s -> memory-bound, ~10% of peak; suspect multi-pass + small BLOCK_SIZE")


# ---------------------------------------------------------------------------
# Puzzle 2: matmul (2048, 2048, 2048) fp16 in 3.5 ms.
# FLOPs = 2 * 2048^3 = 17,179,869,184 ≈ 1.72e10
# Time: 3.5e-3 s
# TFLOP/s = 1.72e10 / 3.5e-3 / 1e12 = 4.91 TFLOP/s
# Peak fp16 tensor-core = ~57 TFLOP/s
# ---------------------------------------------------------------------------
# Diagnosis: 4.9 TFLOP/s is ~9% of peak. Compute-bound territory in principle
# (matmul flop/byte ratio is high), but achieving only 9% means either the
# kernel isn't using tensor cores (no tl.dot, wrong dtype), or BLOCK sizes
# are too small to amortize the K-loop overhead, or num_warps/num_stages are
# wrong.
#
# Next experiment:
#   1) Confirm tl.dot is being used (not a hand-rolled loop).
#   2) Check accumulator is fp32, inputs fp16.
#   3) Autotune over (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps).
#   4) Check that the grid produces enough programs to fill the SMs:
#      cdiv(2048, BLOCK_M) * cdiv(2048, BLOCK_N) >> num_SMs (40 on 2070 SUPER).
def puzzle_2():
    flops = 2 * 2048 ** 3
    ms = 3.5
    tfs = flops / (ms * 1e-3) / 1e12
    print(f"puzzle 2: {tfs:.2f} TFLOP/s -> compute-bound in theory; only 9% of peak, autotune the configs")


# ---------------------------------------------------------------------------
# Puzzle 3: W8A16 decode matmul (1, 4096, 4096) in 0.18 ms.
# Bytes (weight dominates):
#     int8 weight:   4096 * 4096 = 16,777,216
#     fp16 act:      4096 * 2    = 8,192
#     fp16 out:      4096 * 2    = 8,192
#     total ≈ 16,793,600 bytes
# Time: 0.18 ms = 1.8e-4 s
# GB/s = 1.679e7 / 1.8e-4 / 1e9 = 93.3 GB/s
# Peak DRAM: ~448 GB/s.
# ---------------------------------------------------------------------------
# Diagnosis: 93 GB/s = ~21% of peak. This kernel is weight-bandwidth-bound
# (the int8 weight matrix is by far the biggest tensor in the equation).
# At only 21% we have headroom.
#
# Next experiment:
#   1) Check coalesced access: the K-axis stride on the weight should be the
#      contiguous one for the loads.
#   2) Larger BLOCK_K and num_warps to overlap more outstanding loads.
#   3) On Ampere+, increase num_stages for software pipelining.
#   4) Verify the dequant multiply is in fp16 (not fp32) so it doesn't slow
#      down the tile.
def puzzle_3():
    bytes_total = 4096 * 4096 * 1 + 4096 * 2 + 4096 * 2
    ms = 0.18
    gbs = bytes_total / (ms * 1e-3) / 1e9
    print(f"puzzle 3: {gbs:.1f} GB/s -> memory-bound; ~21% of peak DRAM, push BLOCK_K and num_warps")


if __name__ == "__main__":
    puzzle_1()
    puzzle_2()
    puzzle_3()
