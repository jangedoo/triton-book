# Chapter 10 Exercises

## Beginner

1. **Plain batched matmul.** Take `batched_matmul_kernel` and strip the
   masks (assume contiguous, power-of-two shapes). Compare against
   `torch.matmul` for `(8, 256, 256) @ (8, 256, 256)`.

2. **Linear without bias.** Adapt the fused kernel to compute
   `Y = X @ W^T` only. Should be a one-line deletion plus removing the
   bias arg.

3. **Linear with bias (no activation).** Same as above but keep the
   bias add. This is the kernel `F.linear` represents.

## Intermediate

4. **Fused linear + bias + GELU.** Walk through `linear_bias_gelu_kernel`
   and reimplement it from scratch. Confirm against
   `F.gelu(F.linear(x, w, b), approximate="tanh")`.

5. **Fused linear + bias + residual.** `Y = X @ W^T + b + R` where R has
   the same shape as Y. Load R alongside the bias in the epilogue, add
   it in fp32, cast, store. Useful for transformer FFN output
   projections where the residual add is the next op.

## Advanced

6. **Why fused linear + LayerNorm is harder.** Write a short paragraph
   explaining what goes wrong if you try to fuse a LayerNorm onto a
   matmul. Hints: LayerNorm needs the mean and variance of the *full*
   hidden-dim vector for each row; the matmul tile only produces
   `BLOCK_N` columns at a time, not the whole row. Sketch what a
   two-pass approach would look like and why most implementations
   leave LayerNorm as a separate kernel.
