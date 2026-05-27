# Chapter 6 exercises

## Beginner

1. **fp32-only kernel first.** Strip the input-dtype casting; assert `x.dtype == torch.float32` in the launcher and remove the per-element `.to(tl.float32)` calls in the kernel. Verify correctness, then add the fp16 path back. The point is to feel the difference: the cast is the only thing keeping you safe at fp16.
   - Hint: try the fp16 version of the test with the cast removed and watch which shapes start failing.

2. **Affine vs non-affine.** Add an `affine: bool` knob to the launcher. When `affine=False`, do not multiply by `weight` or add `bias`. Use a `tl.constexpr` flag rather than branching on a runtime int.
   - Hint: `if AFFINE:` inside the kernel body, where `AFFINE: tl.constexpr` is a kernel argument.

3. **`elementwise_affine=False`.** Same as exercise 2 but match `nn.LayerNorm(..., elementwise_affine=False)`'s semantics — no weight or bias tensors at all. The launcher should accept `weight=None, bias=None`.

## Intermediate

4. **Arbitrary normalized shape.** Support `normalized_shape = (D1, D2)` by reshaping to `(M, D1*D2)` in the launcher. The kernel stays the same; the launcher just does the flatten and unflatten.
   - Hint: compute `H = prod(normalized_shape)` and `M = numel(x) // H`. Be careful that the normalized dims are the *trailing* dims (PyTorch's convention).

5. **Backward kernel.** The chapter ships a backward (see `layernorm.py`). Re-derive it from the math, then implement an alternative split: instead of two kernels (dx + dw/db), do one kernel that computes dx and accumulates dw/db into a per-program buffer via `tl.atomic_add`. Benchmark both.
   - Hint: the dx formula is `dx = (w*dy - xhat * c1 - c2) * rstd` where `c1 = mean(xhat * w*dy)` and `c2 = mean(w*dy)`, both per-row.

## Advanced

6. **Fused dropout + LayerNorm.** Implement a single kernel that applies dropout to `x`, then computes LayerNorm on the dropped tensor. Save the dropout mask for the backward. Compare end-to-end against `F.dropout` then `F.layer_norm` in eager and in `torch.compile`. Note that the dropout *changes the mean and variance* the LayerNorm sees, so this is a genuine numerical change, not just a fusion.
   - Hint: use Philox-style RNG via `tl.rand` seeded by row index. The mask is `tl.rand(...) > p`.
