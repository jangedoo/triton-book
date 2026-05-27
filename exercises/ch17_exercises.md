# Chapter 17 Exercises — KV Cache

Worked solutions are in `solutions/ch17_solutions.py`.

## Beginner

### 1. Write the append kernel

Use the skeleton in `src/ch17_kv_cache/append_kv_cache.py`. Test by writing
17 random tokens one at a time and asserting the first 17 slots of the cache
equal the input.

*Hint:* the new K and V are `[B, H, 1, D]`. The cache slot is at a single
sequence index. One program per `(b, h)` is enough.

### 2. Contiguous-gather kernel

Given `k_cache[B, H, max_seq, D]` and `current_len`, copy
`k_cache[:, :, :current_len, :]` into a fresh contiguous
`[B, H, current_len, D]` buffer. This is *not* how production does it but
writing the kernel makes Exercise 6 easier.

*Hint:* one program per `(b, h, BLOCK_S)`. Mask `offs_s < current_len`.

### 3. Append throughput by batch size

Benchmark `append_kv_cache` across batch sizes `B in (1, 4, 16, 64)` for
`H = 32, D = 128, dtype = fp16`. At which batch size does the kernel
become bandwidth-bound vs launch-overhead-bound?

*Hint:* bytes moved scale with `B`; the kernel launch is fixed. The
cross-over is roughly where `bytes / DRAM_bandwidth > launch_overhead`. On
an A100 launch overhead is ~5 us.

## Intermediate

### 4. GQA decode

Extend `decode_attention` to Group-Query Attention. `Q` has `H_q` heads;
the cache has `H_kv < H_q` heads, with each KV head shared by
`H_q / H_kv` query heads. Two ways to do this:

- (a) Easy: map each query head to its KV head, run as before. Wasteful —
  each KV head gets re-read `H_q / H_kv` times.
- (b) Better: keep one program per `(b, h_kv)`. Process all
  `H_q / H_kv` query heads that share the KV head, accumulating into
  separate outputs. Reads the cache once per group.

Implement (b).

*Hint:* `h_kv = h_q // (H_q / H_kv)`. KV pointer arithmetic uses `h_kv`;
Q pointer arithmetic uses `h_q`. Inside the K-loop, run the inner online
softmax once per `h_q` in the group.

### 5. Block-table lookup primitive

Given `block_table[B, max_blocks]` and a list of logical positions,
return the physical-position indices

```
phys[b, pos] = block_table[b, pos // BLOCK_TOKENS] * BLOCK_TOKENS
             + pos % BLOCK_TOKENS
```

per `(b, pos)`. Just the lookup, no attention.

*Hint:* this is a gather. One program per `(b, BLOCK_POS)`. Divide and
mod by the `tl.constexpr` `BLOCK_TOKENS`.

## Advanced

### 6. Tiny paged-decode attention

Build a paged-decode kernel:
- `Q[B, H, 1, D]`
- paged `K_cache, V_cache` of shape `[num_blocks, BLOCK_TOKENS, H, D]`
- `block_tables[B, max_blocks]`
- `current_lens[B]`

One program per `(b, h)`. Stream over blocks using the block table.

Verify against a contiguous-cache reference: build a contiguous cache from
the paged one as a host-side gather, then run the standard
`decode_attention` and compare.

*Hint:* this combines Exercises 2 and 5. First do a host-side gather to
validate your block-table indexing, then push the lookup inside the
attention kernel. The inner loop becomes:

```
for block_idx in 0..num_blocks_for_this_request:
    phys = block_tables[b, block_idx]
    k_tile = load from K_cache[phys, :, h, :]   # [BLOCK_TOKENS, D]
    ...
```
