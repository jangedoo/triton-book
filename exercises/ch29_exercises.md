# Chapter 29 Exercises

The block is a stage. Take pieces off the stage one at a time.

## Beginner

### Exercise 1: Swap just RMSNorm

Start from `PyTorchBlock`. Replace only `_rmsnorm_torch` with
`mini_triton_llm.rmsnorm`. Keep PyTorch attention and PyTorch SwiGLU.
Confirm correctness, then benchmark forward at `B=4, S=1024`. Report
the speedup (or, more likely, the wash) and explain it in two sentences.

### Exercise 2: Swap just SwiGLU

Same setup. Replace only the `F.silu(a) * b` line with
`mini_triton_llm.swiglu`. The win here is bigger than (1) because the
intermediate dim is large; you save a full bandwidth pass over the
gate tensor.

### Exercise 3: Swap just attention

Replace only the SDPA call with `mini_triton_llm.flash_attention`. This
is the one where the *educational* Triton kernel is likely to lose to
PyTorch (which dispatches to a production FlashAttention). Report both
numbers honestly. The lesson is that production-quality attention is its
own multi-year project.

## Intermediate

### Exercise 4: Build a 4-layer stack

Stack four `TritonBlock`s and time forward end-to-end at `S=1024`.
Compare to four `PyTorchBlock`s. The fusion wins should compound across
layers; the attention loss should also compound. Net it out.

Hint: a four-layer stack will also surface any hidden memory allocations
inside your kernels. Watch the `torch.cuda.max_memory_allocated` curve.

### Exercise 5: Hook up the KV cache

The block as written re-encodes the whole sequence on every call. Add a
KV cache (see Chapter 17): on each forward, take a single new token,
project it to (q1, k1, v1), apply RoPE with `offset=past_len`, append
`k1, v1` to the cached `K, V`, and call `flash_attention` against the
full cached `K, V`. Benchmark decode latency (one new token per call)
for a 1024-token prefix. Compare against an equivalent PyTorch decode.

## Advanced

### Exercise 6: Backward through the Triton block

Run `block(x).sum().backward()` on `TritonBlock`. It will likely fail at
the first Triton op, because we did not register backwards. Fix it.
Choose one of:

1. Wrap each Triton launcher in `torch.autograd.Function` with a Triton
   backward kernel. Re-derive the gradients on paper first.
2. Replace each Triton launcher in the backward path with the PyTorch
   reference. This is cheating but correct; use it to bound the work.

For option 1, write down which kernels need new backward kernels (RMSNorm,
RoPE, SwiGLU, attention) and which are already wired by PyTorch
(everything in `nn.Linear`). For attention, re-read Chapter 15.

Once it runs, gradcheck a tiny configuration (`hidden_dim=16`,
`heads=2`, `S=8`, fp64) against the PyTorch block's backward. Tolerances
will need to be loose; document why.
