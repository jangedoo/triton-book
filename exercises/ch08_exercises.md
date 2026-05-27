# Chapter 8 exercises

## Beginner

1. **GELU exact.** Implement the exact form `0.5 * x * (1 + erf(x / sqrt(2)))`. Compare against `F.gelu(x, approximate="none")`.
   - Hint: `tl.erf` is available. Cast to fp32 first.

2. **SiLU.** Implement `x * sigmoid(x)`. Compare against `F.silu`.
   - Hint: `sigmoid(x) = 1 / (1 + exp(-x))`. Watch the sign on the exponent.

3. **Fused bias + GELU.** Compute `gelu(x + b)` where `b: (D,)` broadcasts along the last dim of `x: (..., D)`. One kernel, two loads (per element of `x`, broadcast for `b`), one store.
   - Hint: pass `x: (N,)` flattened along with `D`, the inner-dim length, and compute `bias_idx = offs % D`. Or do it 2D with one program per row of `x.reshape(-1, D)`.

## Intermediate

4. **SwiGLU.** Implement `silu(x_gate) * x_up`. Two inputs, one output, same shape. The kernel is short; the point is recognizing the pattern: most LLM MLPs are some flavor of "gate it, multiply, project down."
   - Hint: load both, compute silu of the gate, multiply.

5. **Linear + bias + activation (conceptual skeleton).** Write the launcher signature and the kernel skeleton for `y = gelu(x @ W + b)` where `x: (M, K)`, `W: (K, N)`, `b: (N,)`. Do NOT implement the matmul body — leave a `# TODO: matmul accumulator goes here (see Chapter 10)` comment. The point is to think through how the activation slots into the matmul epilogue.
   - Hint: the activation goes inside the kernel right before the `tl.store` of the C tile.

## Advanced

6. **GEGLU.** Implement `gelu(x_gate) * x_up` exactly like SwiGLU but using GELU instead. Benchmark against `F.gelu(g) * u` and `torch.compile(lambda g, u: F.gelu(g) * u)`. Use both the exact and tanh approximations of GELU; report which is faster and whether either Triton variant beats `torch.compile`. Expect it to be a tie at best — these are pure-elementwise kernels and `torch.compile` is excellent at them.

Worked solutions live in `solutions/ch08_solutions.py`.
