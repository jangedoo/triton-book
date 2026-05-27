"""Fused cross-entropy forward and backward over LLM-scale vocabs.

Forward avoids materialising softmax by computing LSE in-kernel and gathering
the target logit in the same row pass. Backward uses saved LSE to recompute
softmax on the fly. See Chapter 20.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def cross_entropy_ref(logits, target, ignore_index=-100, reduction="mean"):
    return F.cross_entropy(
        logits.float(), target, ignore_index=ignore_index, reduction=reduction
    )


@triton.jit
def _ce_forward_kernel(
    logits_ptr, target_ptr, loss_ptr, lse_ptr,
    stride_logits_row,
    N, V, ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V

    t = tl.load(target_ptr + row)
    is_valid = t != ignore_index

    x = tl.load(logits_ptr + row * stride_logits_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    s = tl.sum(tl.exp(x - m), axis=0)
    lse = m + tl.log(s)

    safe_t = tl.where(is_valid, t, 0)
    logit_t = tl.load(logits_ptr + row * stride_logits_row + safe_t).to(tl.float32)

    loss = lse - logit_t
    loss = tl.where(is_valid, loss, 0.0)

    tl.store(loss_ptr + row, loss)
    tl.store(lse_ptr + row, lse)


def cross_entropy_forward(
    logits, target, ignore_index=-100, reduction="mean", return_lse=False,
):
    assert logits.is_cuda and target.is_cuda
    assert logits.dim() == 2 and target.dim() == 1
    N, V = logits.shape

    loss = torch.empty(N, device=logits.device, dtype=torch.float32)
    lse = torch.empty(N, device=logits.device, dtype=torch.float32)
    BS = triton.next_power_of_2(V)
    _ce_forward_kernel[(N,)](
        logits, target, loss, lse,
        logits.stride(0), N, V, ignore_index,
        BLOCK_SIZE=BS,
    )

    if reduction == "none":
        out = loss
    elif reduction == "sum":
        out = loss.sum()
    else:
        num_valid = (target != ignore_index).sum().clamp_min(1)
        out = loss.sum() / num_valid

    if return_lse:
        return out, lse
    return out


@triton.jit
def _ce_backward_kernel(
    dlogits_ptr, logits_ptr, target_ptr, lse_ptr,
    stride_logits_row, stride_dlogits_row,
    N, V, ignore_index, inv_num_valid,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V

    t = tl.load(target_ptr + row)
    is_valid = t != ignore_index
    lse = tl.load(lse_ptr + row)

    x = tl.load(logits_ptr + row * stride_logits_row + cols,
                mask=mask, other=-float("inf")).to(tl.float32)
    p = tl.exp(x - lse)
    one_hot = (cols == t).to(tl.float32)
    grad = (p - one_hot) * inv_num_valid
    grad = tl.where(is_valid, grad, 0.0)
    tl.store(dlogits_ptr + row * stride_dlogits_row + cols, grad, mask=mask)


def cross_entropy_backward(
    logits, target, lse, ignore_index=-100, reduction="mean",
):
    assert logits.is_cuda and target.is_cuda and lse.is_cuda
    N, V = logits.shape
    dlogits = torch.empty_like(logits, dtype=torch.float32)
    if reduction == "mean":
        num_valid = (target != ignore_index).sum().clamp_min(1).item()
        inv = 1.0 / num_valid
    elif reduction == "sum":
        inv = 1.0
    else:
        inv = 1.0  # per-row scale; user multiplies by upstream grad

    BS = triton.next_power_of_2(V)
    _ce_backward_kernel[(N,)](
        dlogits, logits, target, lse,
        logits.stride(0), dlogits.stride(0),
        N, V, ignore_index, inv,
        BLOCK_SIZE=BS,
    )
    return dlogits.to(logits.dtype)
