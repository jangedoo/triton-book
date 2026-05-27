# Learning Triton by Rebuilding Modern LLM Kernels

A mini-book that teaches Triton from first principles by rebuilding the kernels that power modern transformer language models — fused softmax, RMSNorm, matmul, FlashAttention, RoPE, KV cache, dequant matmul, and the rest of the stack you find inside a real inference engine.

## Status

Code, tests, and benchmarks were authored without execution. The author writes from macOS — run all validation on a CUDA box. Numbers cited in benchmark sections are placeholders until a real GPU run replaces them. Treat every kernel as draft-quality until you have personally rerun the tests under `tests/` and the benchmarks under `benchmarks/` on your own hardware.

## What you will learn

- How Triton's programming model maps onto NVIDIA GPUs (program ids, blocks, masks, strides).
- How to write correct, vectorized kernels for the building blocks of a transformer.
- How to fuse multiple operations into a single kernel to cut memory traffic.
- How online softmax and the FlashAttention accumulator avoid materializing the full attention matrix.
- How to write a tiled matmul, then extend it to batched and persistent variants.
- How to handle quantized weights end-to-end (dequant on-the-fly inside matmul).
- How to test kernels against PyTorch references with dtype-appropriate tolerances.
- How to benchmark with `triton.testing.do_bench` and read the numbers honestly.
- How to drive `triton.autotune` without lying to yourself about real-world shapes.
- How to debug kernels when the numbers go wrong (NaNs, masks, strides, accumulation dtype).

## Setup

This repo uses [`uv`](https://github.com/astral-sh/uv) for Python and [Quarto](https://quarto.org/docs/get-started/) for the book itself.

1. Install Python dependencies:
   ```
   uv sync
   ```
2. Install Quarto separately by following the official instructions: <https://quarto.org/docs/get-started/>.
3. On the GPU box, install a CUDA-enabled PyTorch + Triton. `uv sync` pulls a baseline `torch`/`triton`, but you almost certainly want a build that matches your CUDA toolkit — follow the official PyTorch instructions for your platform and let Triton come along as a dependency.
4. Local preview:
   ```
   quarto preview
   ```
5. Static build into `_site/`:
   ```
   quarto render
   ```

## Running tests

On the GPU box:

```
uv run pytest tests/
```

Tests use `torch.testing.assert_close` against PyTorch reference implementations. Tolerances are dtype-appropriate (fp32 tight, fp16/bf16 loose) — the choice is consistent across `tests/` and called out in the chapters that introduce each kernel.

## Running benchmarks

Per-chapter benchmark scripts live in `benchmarks/`. Run one with:

```
uv run python benchmarks/bench_<chapter>.py
```

Each script wraps `triton.testing.do_bench`, warms up, syncs, and reports ms plus an effective bandwidth or throughput number.

## Hardware notes

The author's reference GPU is an NVIDIA RTX 2070 SUPER (Turing, sm_75). That card has solid FP16 and FP32 throughput, has Tensor Cores for FP16 GEMM, but does **not** have:

- FP8 tensor cores (Hopper / Ada / Blackwell only).
- bf16 tensor cores (Ampere+).
- TMA (Tensor Memory Accelerator — Hopper+).
- WGMMA or async warpgroup MMA.

Chapters that touch FP8, TMA, or persistent matmul flag the hardware requirement up front and suggest using a Colab/cloud A100/H100 instance for those experiments. Everything else runs on the 2070 SUPER.

## Project layout

```
.
├── README.md
├── pyproject.toml
├── .python-version
├── _quarto.yml
├── kernel_checklist.md
├── llm_kernel_map.md
├── book/
│   ├── index.qmd
│   ├── _assets/
│   │   ├── README.md
│   │   └── ojs/
│   │       ├── grid-mapping-template.qmd
│   │       └── online-softmax-template.qmd
│   ├── 00-orientation/
│   ├── 01-fundamentals/
│   ├── 02-basic-nn/
│   ├── 03-matmul/
│   ├── 04-attention/
│   ├── 05-llm-fusion/
│   ├── 06-quantization/
│   ├── 07-debug-perf/
│   ├── 08-capstone/
│   └── appendix-study-plan.qmd
├── src/
├── tests/
├── benchmarks/
├── exercises/
└── solutions/
```

## Chapter map

Part 0 — Orientation

- Ch 0 [Why Triton, why now](book/00-orientation/00-why-triton.qmd) — where Triton fits between CUDA and PyTorch and why it matters for LLMs.
- Ch 1 [Setup and first kernel](book/00-orientation/01-setup.qmd) — environment, toolchain, and a vector-add hello world.

Part 1 — Triton Fundamentals

- Ch 2 [The Triton mental model](book/01-fundamentals/02-mental-model.qmd) — programs, blocks, masks, and the relationship to CUDA threads.
- Ch 3 [Memory, pointers, strides](book/01-fundamentals/03-memory-pointers-strides.qmd) — addressing tensors of arbitrary layout from inside a kernel.
- Ch 4 [Reductions and broadcasting](book/01-fundamentals/04-reductions.qmd) — `tl.sum`, `tl.max`, axis semantics, and reduction patterns.

Part 2 — Basic Neural Network Kernels

- Ch 5 [Fused softmax](book/02-basic-nn/05-fused-softmax.qmd) — the canonical "one row per program" kernel.
- Ch 6 [LayerNorm](book/02-basic-nn/06-layernorm.qmd) — mean / variance / affine in one pass.
- Ch 7 [RMSNorm](book/02-basic-nn/07-rmsnorm.qmd) — the modern LLM normalization layer.
- Ch 8 [Activations and SwiGLU](book/02-basic-nn/08-activations-swiglu.qmd) — GELU, SiLU, and the gated MLP path.

Part 3 — Matrix Multiplication

- Ch 9 [Tiled matmul](book/03-matmul/09-matmul.qmd) — the workhorse kernel: tiles, accumulators, masks.
- Ch 10 [Batched matmul and Linear](book/03-matmul/10-batched-matmul-linear.qmd) — bias add, batch dimension, replacing `nn.Linear`.
- Ch 11 [Persistent matmul](book/03-matmul/11-persistent-matmul.qmd) — the SM-resident style favored by modern libraries (Hopper-flavored caveats).

Part 4 — Attention and Transformer Kernels

- Ch 12 [Attention math, end to end](book/04-attention/12-attention-math.qmd) — derivation, shapes, masking, numerical issues.
- Ch 13 [Naive attention in Triton](book/04-attention/13-naive-attention.qmd) — materialize the score matrix on purpose to feel the cost.
- Ch 14 [FlashAttention forward](book/04-attention/14-flashattention-fwd.qmd) — online softmax, tile-streaming, never materializing S.
- Ch 15 [Attention backward](book/04-attention/15-attention-backward.qmd) — recomputation strategy and dQ/dK/dV.
- Ch 16 [RoPE and ALiBi](book/04-attention/16-rope-alibi.qmd) — positional encodings as fused pre-attention transforms.
- Ch 17 [KV cache append and read](book/04-attention/17-kv-cache.qmd) — the inference-time data structure and its kernels.

Part 5 — Modern LLM Layer Fusion

- Ch 18 [Fused residual + RMSNorm](book/05-llm-fusion/18-residual-rmsnorm.qmd) — the most common fusion in modern decoders.
- Ch 19 [Fused bias + activation + gate](book/05-llm-fusion/19-fused-bias-act-gate.qmd) — fusing the SwiGLU path with its bias.
- Ch 20 [Cross-entropy loss](book/05-llm-fusion/20-cross-entropy.qmd) — fused log-softmax + NLL for large vocab.
- Ch 21 [Sampling kernels](book/05-llm-fusion/21-sampling.qmd) — temperature, top-k, top-p as Triton kernels.

Part 6 — Quantization-Oriented Kernels

- Ch 22 [Low-precision data types](book/06-quantization/22-low-precision.qmd) — int8, int4, FP8 layouts, scales, zero points.
- Ch 23 [Dequant matmul](book/06-quantization/23-dequant-matmul.qmd) — on-the-fly dequant fused into the matmul inner loop.

Part 7 — Debugging, Testing, Profiling, Optimization

- Ch 24 [Debugging Triton kernels](book/07-debug-perf/24-debugging.qmd) — `tl.device_print`, masks, the usual suspects.
- Ch 25 [Benchmarking that you can trust](book/07-debug-perf/25-benchmarking.qmd) — `do_bench`, warmup, sync, bandwidth math.
- Ch 26 [Autotuning](book/07-debug-perf/26-autotuning.qmd) — `triton.autotune` configs and the search space.
- Ch 27 [Performance tuning](book/07-debug-perf/27-perf-tuning.qmd) — occupancy, registers, shared memory, vectorization.

Part 8 — Capstone Projects

- Ch 28 [A mini Triton kernel library](book/08-capstone/28-mini-library.qmd) — package the kernels you have written into a clean API.
- Ch 29 [Drop-in replacement for transformer pieces](book/08-capstone/29-replace-transformer-pieces.qmd) — swap your kernels into a real PyTorch model.
- Ch 30 [Kernel design playbook](book/08-capstone/30-design-playbook.qmd) — the repeatable process for taking on the next kernel.

Appendix

- [Study plan](book/appendix-study-plan.qmd) — pacing suggestions and study tracks.

## Spec coverage

### Must implement fully

| Kernel / concept | Chapter | Source path | Tests | Bench |
|---|---|---|---|---|
| Vector add | Ch 1 | `src/ch01_setup/sanity_check.py` | covered by Ch 2 tests | covered by Ch 2 bench |
| Vector add (dissected) | Ch 2 | `src/ch02_mental_model/vector_add.py` | `tests/test_ch02_vector_add.py` | `benchmarks/bench_ch02_vector_add.py` |
| Copy (memory + masks) | Ch 3 | `src/ch03_memory/copy.py` | `tests/test_ch03_memory.py` | `benchmarks/bench_ch03_memory.py` |
| Row sum | Ch 4 | `src/ch04_reductions/row_sum.py` | `tests/test_ch04_reductions.py` | `benchmarks/bench_ch04_reductions.py` |
| Row max | Ch 4 | `src/ch04_reductions/row_max.py` | `tests/test_ch04_reductions.py` | `benchmarks/bench_ch04_reductions.py` |
| Softmax (stable + online) | Ch 5 | `src/ch05_softmax/stable_softmax.py`, `src/ch05_softmax/online_softmax.py` | `tests/test_ch05_softmax.py` | `benchmarks/bench_ch05_softmax.py` |
| LayerNorm forward | Ch 6 | `src/ch06_layernorm/layernorm.py` | `tests/test_ch06_layernorm.py` | `benchmarks/bench_ch06_layernorm.py` |
| RMSNorm forward | Ch 7 | `src/ch07_rmsnorm/rmsnorm.py` | `tests/test_ch07_rmsnorm.py` | `benchmarks/bench_ch07_rmsnorm.py` |
| Residual add + RMSNorm | Ch 18 | `src/ch18_residual_rmsnorm/residual_rmsnorm.py` | `tests/test_ch18_residual_rmsnorm.py` | `benchmarks/bench_ch18_residual_rmsnorm.py` |
| GELU / SiLU | Ch 8 | `src/ch08_activations/gelu.py`, `src/ch08_activations/silu.py` | `tests/test_ch08_activations.py` | `benchmarks/bench_ch08_activations.py` |
| SwiGLU forward | Ch 8 / Ch 19 | `src/ch08_activations/swiglu.py`, `src/ch19_fused_swiglu/swiglu_bias.py` | `tests/test_ch08_activations.py`, `tests/test_ch19_fused_swiglu.py` | `benchmarks/bench_ch08_activations.py`, `benchmarks/bench_ch19_fused_swiglu.py` |
| Basic matmul (tiled) | Ch 9 | `src/ch09_matmul/naive_matmul.py`, `src/ch09_matmul/grouped_matmul.py` | `tests/test_ch09_matmul.py` | `benchmarks/bench_ch09_matmul.py` |
| Linear + bias | Ch 10 | `src/ch10_batched_linear/linear_bias_gelu.py` | `tests/test_ch10_batched_linear.py` | `benchmarks/bench_ch10_batched_linear.py` |
| RoPE forward | Ch 16 | `src/ch16_rope/rope.py` | `tests/test_ch16_rope.py` | `benchmarks/bench_ch16_rope.py` |
| Cross-entropy forward (+ logsumexp) | Ch 20 | `src/ch20_cross_entropy/cross_entropy.py`, `src/ch20_cross_entropy/logsumexp.py` | `tests/test_ch20_cross_entropy.py` | `benchmarks/bench_ch20_cross_entropy.py` |
| Simplified FlashAttention forward (causal, D=64) | Ch 14 | `src/ch14_flashattention/flash_attn_fwd.py` | `tests/test_ch14_flashattention.py` | `benchmarks/bench_ch14_flashattention.py` |

### Explained deeply, implementation optional or partial

| Kernel / concept | Chapter | Source path | Notes |
|---|---|---|---|
| Attention backward | Ch 15 | `src/ch15_attention_backward/educational_backward.py` | Educational reference, not production-grade. |
| Persistent matmul | Ch 11 | `src/ch11_persistent_matmul/persistent_matmul_skeleton.py` | Skeleton + concept; full version needs Hopper. |
| FP8 / FP4 block-scaled matmul | Ch 22 | `book/06-quantization/22-low-precision.qmd` | Conceptual chapter; FP8 tensor cores need Hopper+. |
| Paged attention | Ch 17 | `book/04-attention/17-kv-cache.qmd` (concept) | Plain KV cache implemented; paged variant is an exercise. |
| Quantized matmul (W8A16) | Ch 23 | `src/ch23_quant/w8a16_matmul.py`, `src/ch23_quant/dequant.py` | Dequant-in-loop reference. |
| Top-p sampling | Ch 21 | `src/ch21_sampling/top_k.py`, `src/ch21_sampling/temperature.py` | Top-k + temperature implemented; top-p discussed but not implemented. |
| Production-grade transformer block fusion | Ch 28 / Ch 29 | `src/ch28_mini_lib/mini_triton_llm/`, `src/ch29_transformer_swap/triton_block.py` | Capstone-level integration; not a single fused mega-kernel. |

### Gaps

- **Top-p / nucleus sampling kernel** is conceptual only. The `src/ch21_sampling/` module ships `argmax`, `temperature`, and `top_k`; top-p is left as an exercise per Ch 21 ex 3.
- **Paged attention** is described in Ch 17 but only `append_kv_cache.py` and `decode_attention.py` ship. The paged-KV variant is left as an exercise.
- **Attention backward** ships an educational, not production, reference (`educational_backward.py`). Matches the spec ("explained deeply, optional"), so this is a soft gap.
- **Persistent matmul** ships a skeleton. The full Hopper version is hardware-gated and not the target reader's GPU.
- **FP8 / FP4 block-scaled matmul** is conceptual only (Ch 22). Same hardware-gating reason.

None of the "Must implement fully" rows are missing source paths.
