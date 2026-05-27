# Chapter 7 exercises

## Beginner

1. **Forward.** Implement the forward kernel from scratch (without peeking at `src/ch07_rmsnorm/rmsnorm.py`). Verify against `_ref_rmsnorm` in the test module on hidden sizes 768, 1024, 2048, 4096, 8192. Make sure the fp16 test passes with `atol=1e-2`.
   - Hint: compute `mean(x*x)` in fp32, take `rsqrt(mean_sq + eps)`, multiply.

2. **No-gamma (pure RMS).** Add a `WEIGHTED: tl.constexpr` flag. When `False`, skip the multiplication by `w`. Compare against `x * rstd`.
   - Hint: the launcher accepts `weight=None`; pass a dummy 1-element tensor to satisfy the kernel signature.

3. **eps as kernel arg vs constexpr.** Write two versions: one where `eps` is a runtime `float` argument, one where it is `tl.constexpr`. Benchmark both. The constexpr version may be marginally faster because the constant folds into the `rsqrt`.
   - Hint: declaring `eps` as `tl.constexpr` means recompiling for every distinct value. Fine if you only ever use one eps.

## Intermediate

4. **Backward kernel.** Re-derive the backward from the math in the chapter, then implement it. Compare against PyTorch autograd on fp32 inputs with `gradcheck`-style tolerances.
   - Hint: $dx_j = r (w_j dy_j) - \hat{x}_j \cdot \tfrac{1}{H} \sum_i \hat{x}_i w_i dy_i$ with $\hat{x} = x \cdot r$.

5. **In-place RMSNorm.** Write `rmsnorm_inplace_(x, w, eps)` that reuses `x`'s storage for the output. Useful when memory is tight (preview of Ch 18). Watch out: the kernel reads `x` and writes `x`, so the writes must not happen until after the per-row reduction.
   - Hint: it is safe because the whole row is in registers before the first store. Just pass `y_ptr = x_ptr`.

## Advanced

6. **Fused residual-add + RMSNorm.** Compute `y = rmsnorm(x + residual, w)` in one kernel. This is the kernel Llama et al. call between every block; Chapter 18 covers it formally, but try to write it yourself first. Also return the post-add tensor (`x + residual`) because the next residual stream needs it.
   - Hint: two input pointers, one extra load, one extra store. The `rstd` is computed on the *sum*, not on `x` alone. Benchmark and compare against two separate kernels.
