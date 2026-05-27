# Chapter 3 Exercises

Six exercises on memory, pointers, strides, and masks. Solutions live in
`solutions/ch03_solutions.py`.

## Beginner

### B1. Strided 1-D copy

Modify `copy_kernel` so it accepts an input stride and an output stride and
copies through them. Verify against `x[::2].clone()` for a tensor of length
`2 * N`.

*Hint:* this is essentially Exercise A1 from Chapter 2 with no addition.

### B2. Row-wise scale

Write a kernel `row_scale(x, scale)` where `scale` is a 1-D tensor of length
`M` (one factor per row). One program per row. Compare against `x * scale[:, None]`.

*Hint:* `scale[pid_m]` is a scalar load with no offset vector — you do not
need `tl.arange` for it.

### B3. Column-wise add

Like `row_add` but the bias has length `M` and you add `bias[i]` to row `i`.
One program per row, inner loop along columns, `bias_value` is a single
scalar inside the kernel.

*Hint:* call `tl.load(bias_ptr + pid_m)` once, before the column loop.

## Intermediate

### I1. Strided `vector_add`

Write `vector_add_strided(x, y)` that handles 1-D non-contiguous tensors on
both inputs *and* the output. Confirm correctness when `x = full[::2]` and
`y = full[1::2]`. Measure how much bandwidth you lose versus the contiguous
case.

*Hint:* the only thing that changes versus Chapter 2 is multiplying every
offset by the appropriate stride.

### I2. In-place row add

Modify `row_add` so it writes back into `x` instead of returning a fresh
tensor. Make sure the test still passes when you pass `x.clone()` as the
reference baseline.

*Hint:* you do not need a new output pointer — pass `x_ptr` twice and reuse
it for the store.

## Advanced

### A1. Strided 2-D copy that handles arbitrary memory layouts

Write `copy2d(x)` that returns a contiguous copy of any 2-D `x` (including
the output of `.t()`, `[:, ::2]`, or broadcast views). Pass strides for both
input and output. Compare your bandwidth to `x.contiguous()` for at least
three layouts: contiguous, transposed, and column-strided.

*Hint:* the kernel from `transpose.py` is the structural template. You are
just removing the swap on the output side.
