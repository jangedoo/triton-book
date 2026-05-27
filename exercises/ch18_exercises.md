# Chapter 18 — Exercises

## Beginner

**1. Vanilla return.** Strip the `STORE_X_NEW` path so the kernel returns only `y`. Confirm by setting the `x_new` pointer to `None` in the launcher when not needed. Hint: a `tl.constexpr` flag lets the compiler erase the store entirely.

**2. Both outputs.** Use the chapter kernel to return `(y, x + residual)`. Write a test that materialises `x + residual` in fp32 with PyTorch and confirms exact bitwise equality with the kernel's `x_new` output in fp32 (no rounding error because there is no reduction on this path).

**3. No gamma.** Run the kernel with `weight=None`. Confirm the output equals the kernel run with `weight=torch.ones(N)`. Hint: use the `HAS_WEIGHT` constexpr.

## Intermediate

**1. Drop-residual flag.** Add a `drop_residual: bool` constexpr. When true, the kernel skips the residual load and the add — it becomes plain RMSNorm. Useful for the first transformer block which has no residual to carry yet.

**2. Fused dropout.** Insert dropout between `x + residual` and the variance reduction. Use `tl.rand(seed, offsets)` to generate the mask. Scale by `1 / (1 - p)`. Test against PyTorch with matched seeds.

## Advanced

**1. Backward kernel.** Derive `dx`, `dresidual`, and `dweight` analytically — note that both `dx` and `dresidual` are equal up to dtype. Implement a one-program-per-row backward kernel. Use `gradcheck` on small fp32 inputs.
