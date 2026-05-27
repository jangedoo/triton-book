# Chapter 4 Exercises

Six exercises on reductions. Solutions live in `solutions/ch04_solutions.py`.

## Beginner

### B1. Row product

Write `row_prod(x)` that computes the row-wise product. Multiplicative
identity is `1.0` — initialize the accumulator and the `other=` argument
accordingly.

*Hint:* `tl.reduce(x, axis=0, combine_fn=mul)` is overkill; you can do
`acc *= tl.prod(tile)` (use `tl.reduce` with `tl.multiply` if your Triton
version lacks `tl.prod`).

### B2. Row L2 norm

Write `row_l2(x)` that computes `sqrt(sum(x ** 2))` per row. Build it from
`row_sum` of `x * x` with a final `tl.sqrt` on the scalar.

*Hint:* the masked load must use `other=0.0` so the masked lanes contribute
nothing to the sum of squares.

### B3. Row argmax index only

Write `row_argmax(x)` that returns the *index* of the maximum along each row
(no value). You will need to keep a running pair `(best_val, best_idx)` and
update both whenever a new lane beats the running best.

*Hint:* `tl.where(new > best_val, ...)` for both fields, in lockstep.

## Intermediate

### I1. Column-wise sum

Write `col_sum(x)` that computes `x.sum(dim=0)` for an `[M, N]` input. One
program per column-tile. Each program loops over rows and accumulates a
`(BLOCK_SIZE_N,)` vector.

*Hint:* the accumulator is now a vector, not a scalar. `tl.zeros((BLOCK_SIZE_N,), tl.float32)`.

### I2. Numerically stable LogSumExp per row

Write `row_logsumexp(x)`. Use the max-subtraction trick: compute the row
max, then `log(sum(exp(x - max))) + max`. Two passes: pass 1 finds the max,
pass 2 accumulates the shifted exps. This is one step away from softmax —
the next chapter writes the same control flow.

*Hint:* `tl.log` exists. Promote the intermediate `exp` accumulator to
fp32 even for fp16 input.

## Advanced

### A1. One-pass Welford variance

Replace the two-pass `row_variance` with the [Welford online
algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
to compute variance in a single read of the row. Carry `(count, mean, M2)`
through the column loop. Verify it matches the two-pass kernel on a
shifted-mean input (`x + 1e6`) where the naive `E[X^2] - E[X]^2` would
collapse.

*Hint:* the per-tile combine step is non-trivial because each tile already
has its own local `(n, mean, M2)`. Look up "parallel Welford" — the update
formula for combining two partials is in the Wikipedia article.
