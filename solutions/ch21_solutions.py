"""Solutions for Chapter 21 exercises."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ch21_sampling import temperature_scale, argmax_sample, top_k_mask


# Exercise 1 (Beginner): temperature scale — chapter kernel.
def temp_solution(x, T):
    return temperature_scale(x, T)


# Exercise 2 (Beginner): greedy argmax — chapter kernel.
def argmax_solution(x):
    return argmax_sample(x)


# Exercise 3 (Beginner): top-k mask — chapter kernel.
def topk_solution(x, k):
    return top_k_mask(x, k)


# Intermediate 1: fused temperature + top-k + softmax + multinomial draw.
@triton.jit
def _fused_sampler_kernel(
    logits_ptr, out_ptr,
    N, V, stride_row,
    inv_t, k, seed,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V

    x = tl.load(logits_ptr + row * stride_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32) * inv_t

    # find the k-th largest via repeated max-and-suppress; only correct for small k.
    # for production you'd implement a proper top-k; here we approximate by setting
    # the threshold as: kth largest = ... we'll use a sort-by-threshold approach.
    # simplest: compute the row's full softmax, then a one-shot "keep top-k by rank".
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    p = e / tl.sum(e, axis=0)

    # naive top-k via k passes — fine for k <= 64
    thresh = m  # will be overwritten
    for _ in range(k):
        idx = tl.argmax(x, axis=0)
        val = tl.max(x, axis=0)
        thresh = val
        x = tl.where(cols == idx, -float("inf"), x)
    # at this point, thresh is the k-th largest of the original logits
    # (after k argmax-then-suppress rounds)
    p = tl.where(cols >= 0, p, 0.0)  # noop; placeholder
    # Recompute kept softmax using the threshold:
    x_orig = tl.load(logits_ptr + row * stride_row + cols,
                     mask=mask, other=-float("inf")).to(tl.float32) * inv_t
    kept = x_orig >= thresh
    x_kept = tl.where(kept, x_orig, -float("inf"))
    m2 = tl.max(x_kept, axis=0)
    e2 = tl.exp(x_kept - m2)
    p2 = e2 / tl.sum(e2, axis=0)

    # uniform draw via tl.rand
    u = tl.rand(seed, row)
    cum = tl.cumsum(p2, axis=0)
    chosen = tl.sum((cum < u).to(tl.int32), axis=0)
    # clamp to V-1 in case of rounding
    chosen = tl.minimum(chosen, V - 1)
    tl.store(out_ptr + row, chosen.to(tl.int64))


def fused_sample(logits, temperature=1.0, k=50, seed=0):
    """Fused sampler. Only meaningful for small V (k passes are O(k)).

    Returns one sampled token id per row.
    """
    N, V = logits.shape
    assert V <= 4096, "fused sampler is for small vocabs"
    out = torch.empty(N, device=logits.device, dtype=torch.int64)
    BS = triton.next_power_of_2(V)
    _fused_sampler_kernel[(N,)](
        logits, out, N, V, logits.stride(0),
        1.0 / float(temperature), k, seed,
        BLOCK_SIZE=BS,
    )
    return out


# Intermediate 2: top-p with Triton cumsum + mask on already-sorted input.
@triton.jit
def _topp_scan_kernel(
    sorted_logits_ptr, out_ptr,
    N, V, stride_row, p,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(sorted_logits_ptr + row * stride_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    probs = e / s
    cum = tl.cumsum(probs, axis=0)
    # drop where (cum - probs) > p, equivalently cum_prev > p
    drop = (cum - probs) > p
    y = tl.where(drop, -float("inf"), x)
    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


def top_p_triton(logits, p):
    N, V = logits.shape
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_logits = sorted_logits.contiguous()
    out_sorted = torch.empty_like(sorted_logits)
    BS = triton.next_power_of_2(V)
    _topp_scan_kernel[(N,)](
        sorted_logits, out_sorted,
        N, V, sorted_logits.stride(0), p,
        BLOCK_SIZE=BS,
    )
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_idx, out_sorted)
    return out


# Advanced: min-p sampler (Triton-only, single kernel).
@triton.jit
def _min_p_sample_kernel(
    logits_ptr, out_ptr,
    N, V, stride_row,
    min_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(logits_ptr + row * stride_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    probs = e / s
    max_p = tl.max(probs, axis=0)
    keep = probs >= (min_p * max_p)
    probs = tl.where(keep, probs, 0.0)
    probs = probs / tl.sum(probs, axis=0)
    u = tl.rand(seed, row)
    cum = tl.cumsum(probs, axis=0)
    chosen = tl.sum((cum < u).to(tl.int32), axis=0)
    chosen = tl.minimum(chosen, V - 1)
    tl.store(out_ptr + row, chosen.to(tl.int64))


def min_p_sample(logits, min_p=0.05, seed=0):
    N, V = logits.shape
    out = torch.empty(N, device=logits.device, dtype=torch.int64)
    BS = triton.next_power_of_2(V)
    _min_p_sample_kernel[(N,)](
        logits, out, N, V, logits.stride(0),
        min_p, seed, BLOCK_SIZE=BS,
    )
    return out
