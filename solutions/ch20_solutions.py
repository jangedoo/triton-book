"""Solutions for Chapter 20 exercises."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Exercise 1 (Beginner): logsumexp.
from ch20_cross_entropy import logsumexp as chapter_logsumexp


def lse_solution(x):
    return chapter_logsumexp(x)


# Exercise 2 (Beginner): cross-entropy forward without ignore_index.
@triton.jit
def _ce_no_ignore_kernel(
    logits_ptr, target_ptr, loss_ptr,
    stride_row, N, V,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    t = tl.load(target_ptr + row)
    x = tl.load(logits_ptr + row * stride_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    lse = m + tl.log(tl.sum(tl.exp(x - m), axis=0))
    logit_t = tl.load(logits_ptr + row * stride_row + t).to(tl.float32)
    tl.store(loss_ptr + row, lse - logit_t)


def ce_no_ignore(logits, target, reduction="mean"):
    N, V = logits.shape
    loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    BS = triton.next_power_of_2(V)
    _ce_no_ignore_kernel[(N,)](logits, target, loss, logits.stride(0), N, V,
                               BLOCK_SIZE=BS)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# Exercise 3 (Beginner): ignore_index — chapter kernel already supports it.
from ch20_cross_entropy import cross_entropy_forward


def ce_with_ignore(logits, target, ignore_index=-100):
    return cross_entropy_forward(logits, target, ignore_index=ignore_index)


# Intermediate 1: backward — chapter ships it.
from ch20_cross_entropy import cross_entropy_backward


def ce_full(logits, target, ignore_index=-100):
    loss, lse = cross_entropy_forward(logits, target, ignore_index=ignore_index,
                                      return_lse=True)
    grad = cross_entropy_backward(logits, target, lse, ignore_index=ignore_index)
    return loss, grad


# Intermediate 2: label smoothing.
@triton.jit
def _ce_smooth_kernel(
    logits_ptr, target_ptr, loss_ptr,
    stride_row, N, V, ignore_index, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    t = tl.load(target_ptr + row)
    is_valid = t != ignore_index

    x = tl.load(logits_ptr + row * stride_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    lse = m + tl.log(tl.sum(tl.exp(x - m), axis=0))

    # mean over valid lanes only
    mean_x = tl.sum(tl.where(mask, x, 0.0), axis=0) / V

    safe_t = tl.where(is_valid, t, 0)
    logit_t = tl.load(logits_ptr + row * stride_row + safe_t).to(tl.float32)

    ce = lse - logit_t
    smooth = lse - mean_x  # -mean(log_softmax) = LSE - mean(logits)
    out = (1.0 - eps) * ce + eps * smooth
    out = tl.where(is_valid, out, 0.0)
    tl.store(loss_ptr + row, out)


def ce_label_smoothing(logits, target, ignore_index=-100, label_smoothing=0.0):
    N, V = logits.shape
    loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    BS = triton.next_power_of_2(V)
    _ce_smooth_kernel[(N,)](
        logits, target, loss, logits.stride(0),
        N, V, ignore_index, label_smoothing,
        BLOCK_SIZE=BS,
    )
    num_valid = (target != ignore_index).sum().clamp_min(1)
    return loss.sum() / num_valid


# Advanced: chunked-vocab online logsumexp.
@triton.jit
def _online_lse_kernel(
    x_ptr, out_ptr,
    stride_row, N, V,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    running_max = -float("inf")
    running_sum = 0.0
    n_chunks = tl.cdiv(V, BLOCK_V)
    for i in range(n_chunks):
        cols = i * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(x_ptr + row * stride_row + cols,
                    mask=mask, other=-float("inf")).to(tl.float32)
        chunk_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, chunk_max)
        # rescale running_sum to the new max
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max
    lse = running_max + tl.log(running_sum)
    tl.store(out_ptr + row, lse)


def online_logsumexp(x, BLOCK_V=8192):
    N, V = x.shape
    out = torch.empty(N, device=x.device, dtype=torch.float32)
    _online_lse_kernel[(N,)](x, out, x.stride(0), N, V, BLOCK_V=BLOCK_V)
    return out
