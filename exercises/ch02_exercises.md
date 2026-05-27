# Chapter 2 Exercises

Six exercises on the Triton mental model. Three beginner, two intermediate,
one advanced. Solutions live in `solutions/ch02_solutions.py`.

## Beginner

### B1. Vector subtraction

Write `vector_sub(x, y)` that returns `x - y`. Reuse the `vector_add`
structure verbatim and change the single line that computes the body. Confirm
your output matches `x - y` for `N = 1_048_576` in fp32.

*Hint:* the only line you need to touch is the one between the two `tl.load`
calls and the `tl.store`.

### B2. Scalar multiply

Write `scalar_mul(x, alpha)` where `alpha` is a Python float passed as a
regular kernel argument (not `tl.constexpr`). The kernel multiplies every
element by `alpha`.

*Hint:* Triton promotes Python scalars to the dtype of the tensor expression
they appear in. You do not need to materialize `alpha` into a tensor.

### B3. Fused multiply-add

Write `fma(x, y, z)` that computes `x * y + z` in a single kernel pass. Three
loads, one store. Compare its bandwidth against doing `x * y` and then
`+ z` in two separate PyTorch ops.

*Hint:* the GB/s formula changes — you now have `4 * N * dtype_bytes` of
traffic per element. Fix the benchmark formula or you will read garbage.

## Intermediate

### I1. Clamp

Write `clamp(x, lo, hi)`. Use `tl.where` (you will meet it again in Chapter
4) or two `tl.minimum` / `tl.maximum` calls. Make sure your kernel passes the
exact-equality test against `torch.clamp(x, lo, hi)` for fp32 input.

*Hint:* fp32 clamp is exact — use `rtol=0, atol=0` and force yourself to
get it right.

### I2. Conditional copy

Write `where_kernel(cond, x, y)` that returns `x` where `cond` is true and
`y` otherwise. `cond` is a `torch.bool` tensor.

*Hint:* `tl.load` of a bool tensor gives you an `int8`/`int1` value; compare
it to zero or use it directly as a mask depending on your Triton version.

## Advanced

### A1. Strided vector add

Write `vector_add_strided(x, y)` that supports non-contiguous `x` and `y`.
Pass `x.stride(0)`, `y.stride(0)`, and `out.stride(0)` as kernel args, and
compute offsets as `block_start + tl.arange(0, BLOCK_SIZE)` multiplied by the
stride. This is a teaser for Chapter 3 — you do not need a clean abstraction
yet. Just get a single-axis strided case to work and measure how much
bandwidth you lose versus the contiguous version.

*Hint:* create the strided input as `x = torch.randn(N * 2, device="cuda")[::2]`
and confirm `x.is_contiguous() is False`.
