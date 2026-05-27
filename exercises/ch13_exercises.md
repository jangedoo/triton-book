# Chapter 13 exercises: Naive attention

## Beginner

1. **Memory accounting.** Compute the approximate memory footprint of the
   `scores` tensor for `B = 4`, `H = 32`, `S = 4096`, fp16. Confirm the
   ~4 GB number from Chapter 12. Repeat for `S = 8192` and explain why
   it will not run on the author's 8 GB 2070 SUPER. Hint: bytes is
   `B * H * S * S * 2`.

2. **Standalone causal mask kernel.** Take `causal_mask_kernel` and
   wrap it in a script that runs on an arbitrary `[B, H, S, S]` tensor.
   Verify against `torch.triu(...).masked_fill(...)` for `S = 9` (a
   non-power-of-two extent that forces the mask path).

3. **Per-S timing.** Use `triton.testing.do_bench` to time the four-kernel
   pipeline for `S in [512, 1024, 2048]` with `B = 1, H = 4, D = 64`.
   Plot the wall clock on log scale and confirm the quadratic growth.

## Intermediate

4. **Chunked softmax.** The current softmax reads the full
   `[B, H, S, S]` `scores` tensor twice (once for the max, once for the
   normalization). Implement a chunked variant that processes `S` in
   blocks of, say, `256`, storing only `[B, H, S]` per-row max and
   normalizer between the two passes. Confirm correctness against the
   one-shot version. Note: this still materializes the score tensor;
   it just reduces softmax-time memory pressure.

5. **Non-causal variant.** Add a `causal: bool` parameter that, when
   false, skips the mask kernel entirely. Add a non-causal test case
   to `test_ch13_naive_attention.py`.

## Advanced

6. **One-kernel QK^T + softmax.** Write a single Triton kernel that
   computes `softmax((Q @ K^T) * scale, dim=-1)` and stores the result
   directly. Each program owns a `[BLOCK_M, S]` row strip. You will
   need two passes over `K`: once to find the row max, once to compute
   the normalized exp. The output is still `[B, H, S, S]`. This
   foreshadows FlashAttention, which removes the second pass entirely.
   Compare wall clock to the four-kernel pipeline at `S = 1024`.
