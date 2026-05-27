# Chapter 28 Exercises

The library is intentionally small. These exercises grow it.

## Beginner

### Exercise 1: Add a `geglu` kernel

GeGLU is the SwiGLU recipe with GELU instead of SiLU:

```
geglu(a, b) = gelu(a) * b
```

Copy `swiglu.py` to `geglu.py`, swap `tl.sigmoid(a) * a` for an approximate
GELU (`0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a**3)))`), export
it from `__init__.py`, and add a correctness test against
`F.gelu(a) * b`.

Hint: Triton has `tl.math.tanh`. Stay in fp32 for the polynomial.

### Exercise 2: Write a top-level smoke test

Without `pytest`, write a `tests/smoke_import.py` script that imports each
public symbol and prints `"ok"`. This is your CI canary — when the package
breaks because someone reshuffled a module, this script catches it before
the longer kernel tests do.

### Exercise 3: Benchmark two kernels at once

Write `benchmarks/bench_two_kernels.py` that benchmarks `rmsnorm` and
`softmax` at the same shape and prints both rows via
`benchmarking.print_table`. Use the existing `compare` helper.

## Intermediate

### Exercise 4: Add type hints and metadata

Add a top-level `pyproject.toml` next to `mini_triton_llm/` with name,
version, Python requirement (`>=3.9`), and a `torch>=2.1` dependency.
Tighten the type hints in `rmsnorm.py` and `softmax.py` — use
`torch.Tensor` everywhere and `float | int` where appropriate.

Hint: `from __future__ import annotations` is already at the top of each
module, so you can use PEP 604 union syntax even on older Pythons.

### Exercise 5: Add `__version__` and an introspection helper

Add `version.py` with `__version__ = "0.1.0"` and re-export it from
`__init__.py`. Add a `mini_triton_llm.kernels_available()` function that
returns a list of the registered public kernel names by reading
`__all__`. This is the kind of helper a user calls from a notebook to
confirm an upgrade landed.

## Advanced

### Exercise 6: Make one kernel differentiable

Wrap `rmsnorm` in a `torch.autograd.Function` so that calling
`rmsnorm(x, w).sum().backward()` Just Works. You will need:

1. A `forward` that calls the existing Triton kernel and saves `(x, w,
   inv_rms)` for backward.
2. A `backward` that computes `dx` and `dw`. The dx formula is the
   non-trivial one — derive it from `y = w * x * rsqrt(mean(x**2) + eps)`
   and the chain rule.
3. A gradcheck on a tiny fp64 input.

Compare against PyTorch's `nn.RMSNorm` backward. Where does your kernel
beat it? Where does it lose? Write a short paragraph in
`solutions/ch28_solutions.py` summarizing the finding.
