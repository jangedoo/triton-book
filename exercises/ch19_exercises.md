# Chapter 19 — Exercises

## Beginner

**1. Bias + SiLU only.** Implement `silu(x + b)` — no gate, no multiply. The smallest fused variant in this family. Verify against `F.silu(x + b)`.

**2. SiLU * gate, no bias.** Call the chapter kernel with `b_gate=None` and `b_up=None`. Confirm the `HAS_BG=False` / `HAS_BU=False` paths produce correct output.

**3. GEGLU.** Run `geglu_bias` and compare against `F.gelu(g, approximate="tanh") * u`. Important: pass `approximate="tanh"` on both sides of the comparison.

## Intermediate

**1. Optional biases mix-and-match.** Build a test matrix over all four combinations of `(b_gate, b_up)` being `None` or a tensor. Confirm every combination matches the PyTorch reference.

**2. In-place writeback.** Make a launcher variant that writes the output back into `x_up`. Verify that the buffer is correctly overwritten and that no race exists (Triton tile lifetime guarantees the load-then-store ordering within one program).

## Advanced

**1. Backward kernel.** Derive the gradients:

- `dsilu(x)/dx = silu(x) + sigmoid(x) * (1 - silu(x))`
- `d(out)/d(xg + bg) = dy * up * dsilu(xg + bg)`
- `d(out)/d(xu + bu) = dy * silu(xg + bg)`

Implement a single 2D-tile backward kernel. Use atomic adds on the bias gradients across the `M` axis. Validate against `torch.autograd` on small fp32 inputs.
