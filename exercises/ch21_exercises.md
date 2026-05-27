# Chapter 21 — Exercises

## Beginner

**1. Temperature scale.** Implement the temperature kernel and verify that `temperature=1.0` is bit-identical to the input.

**2. Greedy argmax.** Implement `argmax_sample`. Test for single-row (decode) and batched. Confirm dtype is `int64`. Compare against `torch.argmax`.

**3. Top-k mask.** Verify `top_k_mask` for `k = 1, 10, 50, 256`. Confirm that exactly `k` (or more, in the rare-tie case) finite entries survive per row.

## Intermediate

**1. Fused sampler.** Combine temperature + top-k + softmax + multinomial draw into a single kernel for `V <= 4096`. Use `tl.rand(seed, row)` for the uniform variate. Validate by sampling many times and comparing the empirical histogram to the unfused reference.

**2. Top-p with Triton scan.** After `torch.sort` on the host, write a Triton kernel that performs `tl.cumsum` of the softmax and emits the dropped-position mask in one pass. Then scatter back. Compare against the all-PyTorch reference for several `p` values.

## Advanced

**1. Min-p sampler (Triton-only).** Min-p keeps tokens with `prob(t) >= min_p * max_prob`. Unlike top-p, this is row-local: no sort. Write one kernel that:

  - computes the row max,
  - computes softmax in one pass,
  - masks tokens below `min_p * max_prob`,
  - renormalises,
  - samples one token using `tl.rand`.

Validate the distribution by drawing many samples and comparing histograms to a PyTorch reference.
