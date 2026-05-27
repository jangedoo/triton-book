# Chapter 16 Exercises — RoPE and ALiBi

Worked solutions are in `solutions/ch16_solutions.py`.

## Beginner

### 1. Non-interleaved RoPE forward

Implement non-interleaved (LLaMA) RoPE forward as a Triton kernel from
scratch. Use the kernel and launcher in `src/ch16_rope/rope.py` as a
starting reference, but try to write your version without copying.

*Hint:* `D` must be even. Make `HALF = D // 2` a `tl.constexpr`. Two
separate `tl.load`s for `x[..., :HALF]` and `x[..., HALF:]`.

### 2. Interleaved variant

Implement the interleaved (GPT-J) variant. The cache is the same; only the
indexing differs — pair `i` is `(x[2i], x[2i+1])`.

*Hint:* use `2 * offs_h` and `2 * offs_h + 1` for the even and odd pointer
arrays.

### 3. Cached cos/sin vs in-kernel computation

Write a second kernel that computes `theta = pos * base^(-2i/D)` and the
trig functions inside the kernel (use `tl.cos`, `tl.sin`). Benchmark
against the cached version at `S = 1024` and `S = 4096`. Report which
one wins and by how much.

*Hint:* the cache is `[S, D/2]` fp32 — a few hundred KB for typical
`D = 64..128`. Recomputing trades a DRAM read for arithmetic; on modern
GPUs `tl.cos` is cheap.

## Intermediate

### 4. Position offset for decoding

Add a `pos_offset` argument so a single token at absolute position `P`
during decoding rotates correctly without recomputing the full `Q`. Test
that

```
rope(x[:, :, P:P+1], cos, sin, pos_offset=P) ==
rope(x[:, :, :P+1], cos, sin, pos_offset=0)[:, :, P:P+1]
```

*Hint:* the kernel in `src/ch16_rope/rope.py` already supports this — try
re-implementing it yourself. Add `POS_OFFSET` to `offs_s` before indexing
`cos`/`sin`.

### 5. Partial rotary dims

Some models (e.g. early Phi, some StarCoder variants) rotate only the
first `rotary_dim` of each head dim and pass the rest through. Add a
`ROTARY_DIM: tl.constexpr` parameter. Load and rotate `[..., :ROTARY_DIM]`,
copy through `[..., ROTARY_DIM:]`.

*Hint:* the cache shrinks to `[S, ROTARY_DIM / 2]`. The pass-through region
needs no compute — just a copy, or no work at all if you operate
in-place.

## Advanced

### 6. Fuse RoPE into the Q/K projection

Borrow a small linear-layer kernel from Chapter 10. Have it produce a
`[B, S, H, D]` tile of `Q`, rotate the tile in-register with RoPE, then
store. Compare end-to-end against `linear` then `rope` on shape
`[2, 4096, 32, 128]`.

*Hint:* the projection writes its accumulator tile after the K-loop.
Insert the rotation just before the store. Load `cos`/`sin` once per
program at the start. You may have to reshape the output tile from
`[BLOCK_M, BLOCK_N]` to `[BLOCK_M, H, D]` mentally before applying RoPE
to the right axis.
