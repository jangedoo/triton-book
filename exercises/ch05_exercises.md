# Chapter 5 exercises

## Beginner

1. **Causal mask.** Modify `stable_softmax` so that, before the max reduction, columns with index `>= row_idx` are set to `-inf`. Verify it matches `torch.softmax` applied to a manually masked tensor.
   - Hint: pass `row_idx` (the program id) into the mask: `causal = col_offs <= row_idx`.

2. **Temperature scaling.** Add a `tau` argument (Python `float`). The kernel should compute `softmax(x / tau)`. Watch for `tau=0`.
   - Hint: divide `x_f32` by `tau` before subtracting the max.

3. **Log-softmax.** Implement `log_softmax(x)` returning `x - m - log(sum(exp(x - m)))`. Compare against `torch.log_softmax`.
   - Hint: you already have `m` and `denom` in the stable kernel; emit `(x - m) - tl.log(denom)`.

## Intermediate

4. **Variable-length rows.** Take an extra `seq_lens: (M,)` int32 tensor where `seq_lens[i] <= N`. Each program should treat only the first `seq_lens[i]` columns as valid. Padded positions must not contribute to the max or the sum.
   - Hint: load `seq_len = tl.load(seq_lens_ptr + row_idx)` and use it instead of `n_cols` when building the mask.

5. **Column softmax.** Implement softmax along `axis=0` for a 2D input. Discuss why this is harder than row-wise softmax in Triton.
   - Hint: one program per column means strided loads (`row_idx * row_stride + col_idx`); the reduction now runs over the row dimension.

## Advanced

6. **Two-pass vs single-pass online softmax.** The kernel in `online_softmax.py` makes two passes over the row. Implement a single-pass variant that maintains both the running max `m` and a running numerator that gets rescaled by `alpha = exp(m_old - m_new)` on every tile, plus the final divide. Benchmark both on `M=2048, N=32768` in fp16. Report wall-clock and the difference in numerical error vs `torch.softmax`.
   - Hint: the single-pass variant has to write the unscaled `exp(x - m_new)` somewhere; either keep it in registers (only works if the whole row fits) or stage it through global memory (defeats the purpose). Most practical "single-pass online softmax" implementations actually only work when the row fits in one tile — at which point you might as well use `stable_softmax`. Discuss this tradeoff.
