# Chapter 14 exercises: FlashAttention forward

## Beginner

1. **Non-causal variant.** The reference kernel takes `IS_CAUSAL` as a
   constexpr. Verify that setting it to `False` matches
   `F.scaled_dot_product_attention(..., is_causal=False)` for
   `B=1, H=2, S=128, D=64`. Why is the loop length now `S` regardless
   of `pid_m`?

2. **Trace the state by hand.** Run the kernel on the smallest non-trivial
   shape (`B=1, H=1, S=4, D=2`, `BLOCK_M=2, BLOCK_N=2`) and print
   `m_i, l_i, acc` after each K/V block by adding `tl.device_print`
   inside the loop. Compare against a hand-computed reference using
   the online softmax recurrence from the chapter widget.

3. **Block-size sweep.** Time the kernel for
   `(BLOCK_M, BLOCK_N) in {(32, 32), (64, 64), (128, 64), (64, 128)}`
   on `S = 2048, D = 64`. Note which combination is fastest and why
   you might expect that — relate to register pressure (D-dependent)
   and SRAM occupancy.

## Intermediate

4. **Arbitrary head_dim via padding.** Modify the launcher to round `D`
   up to the next power of two, pass `BLOCK_D = next_pow_2(D)`, and
   mask the load with `offs_d[None, :] < D, other=0.0` for `Q`, `K`,
   `V`. Verify correctness for `D = 48` (a non-power-of-two head dim
   used by some GQA configurations).

5. **Temperature scalar.** Add a `temperature: float` argument. Replace
   `scale = 1 / sqrt(D)` with `scale = 1 / (sqrt(D) * temperature)`.
   Verify that `temperature = 2.0` produces a softer attention
   distribution by inspecting `probs` for a small shape.

## Advanced

6. **Variable-length sequences.** Real serving stacks pack multiple
   sequences into one batch with a `seq_lens: int32[B]` tensor.
   Modify the kernel so each `(batch, head)` slice respects its own
   `seq_lens[b]` — positions beyond `seq_lens[b]` are masked to
   `-inf`. Hint: pass `seq_lens_ptr` and load `cur_len = tl.load(seq_lens_ptr + b)`.
   Then `mask_n = (offs_n < cur_len) & (offs_n < S)`, and similarly
   for `mask_m`. For causal mode, the upper-triangle mask also
   depends on `cur_len`. A partial solution lives in
   `solutions/ch14_solutions.py`; the rest is yours.
