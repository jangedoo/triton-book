# Chapter 15 Exercises — Attention Backward

The exercises here are lighter than other chapters and lean on derivations
rather than throughput. Worked solutions are in `solutions/ch15_solutions.py`.

## Beginner

### 1. Re-derive the softmax-row Jacobian

Start from `p_i = exp(s_i) / sum_j exp(s_j)`. Differentiate `p_i` with respect
to `s_j` for both `i = j` and `i != j`. Show that the upstream-pulled gradient
simplifies to `dS_i = P_i * (dP_i - rowsum(dP_i * P_i))`.

*Hint:* factor `P_i` out of the diagonal and off-diagonal cases separately.

### 2. Implement just dV as a kernel

Given a pre-computed `P` of shape `[B, H, S, S]`, write a Triton kernel that
returns `dV = P^T @ dO`. Start from the `_dv_kernel` in
`src/ch15_attention_backward/educational_backward.py` and remove the launcher
clutter.

*Hint:* this is the simplest of the three gradients. Tile `(N, D)`, sweep `M`,
load `P[m, n]`, transpose with `tl.trans`, multiply by `dO[m, d]`.

### 3. FLOPs vs bytes for recompute

For `B=1, H=16, S=4096, D=64`, estimate:
- the memory cost (bytes) of storing `P` as fp32,
- the FLOP cost of recomputing `P` once during backward (`Q @ K^T` plus the
  softmax).

Which approach uses more, and by what ratio?

*Hint:* `P` is `B * H * S * S * 4` bytes. Recomputation is roughly
`2 * B * H * S * S * D` FLOPs for the matmul; softmax is a small additional
term.

## Intermediate

### 4. dQ recomputation skeleton

Write a `_dq_kernel` that, given pre-computed `dS`, returns
`dQ = dS @ K * scale`. Mirror the `_dv_kernel` structure but loop in the
K-dimension and do **not** transpose. Don't worry about online recompute of
`P` — assume `dS` is given.

*Hint:* one program per `(b, h, BLOCK_M)`. The accumulator has shape
`(BLOCK_M, BLOCK_D)`. The loop variable plays the role of the "scan"
dimension.

### 5. Read the tutorial's _attn_bwd_preprocess

Open the official Triton tutorial at `python/tutorials/06-fused-attention.py`.
Locate `_attn_bwd_preprocess`. In one paragraph explain:
- what tensor it computes (give its shape and dtype),
- why precomputing it makes both `_attn_bwd_dq` and `_attn_bwd_dkdv` cheaper.

*Hint:* there is an identity `rowsum(dP * P) = rowsum(dO * O)`. Which side of
that identity uses inputs you already have on hand?

## Advanced

### 6. Causal backward

Extend `attention_backward_educational` to accept `causal=True`. The forward
causal mask zeros the upper triangle of `S` before softmax. The backward has
to honor that mask everywhere.

*Hint:* for `dV[n]` and `dK[n]` the contributing `Q` rows are only those at
positions `m >= n`. For `dQ[m]` the contributing `K`/`V` rows are only those
at positions `n <= m`. You can either skip the masked tiles entirely (faster)
or multiply by a mask inside the inner loop (simpler).
