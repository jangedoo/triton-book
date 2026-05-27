# Chapter 9 Exercises

## Beginner

1. **Matmul + bias.** Take `matmul_naive` and add `bias: (N,)` that gets
   broadcast-added to the accumulator before the final cast. Hint: load it
   once per tile with `b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)`
   and do `acc += b[None, :]`.

2. **Matmul + ReLU epilogue.** Same as above but with
   `acc = tl.maximum(acc, 0.0)` before the store. No new memory traffic
   compared to a separate ReLU kernel.

3. **Transposed A support.** Add an `a_transposed: bool` flag (Python-side)
   and swap which stride is "row-stride" vs "k-stride" in the launcher. The
   kernel itself does not change -- the trick is purely in the launch
   arguments.

## Intermediate

4. **Matmul + GELU epilogue.** Use the tanh approximation:
   `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`. Compare
   against `F.gelu(F.linear(...))` for correctness.

5. **Split-K.** Chunk the K axis across multiple programs and use
   `tl.atomic_add` to accumulate into C. Worth it for skinny matmuls (M, N
   small, K large) where the regular tiling under-utilizes SMs. Implement
   a small version or write pseudocode if hardware time is tight.

## Advanced

6. **Roll your own super-grouping.** Without looking at
   `grouped_matmul.py`, derive the `pid -> (pid_m, pid_n)` mapping for
   GROUP_SIZE_M = 4 by hand on a 8x8 tile grid, then code it up. The point
   is to internalize *why* the indexing dance gives you L2 reuse.
