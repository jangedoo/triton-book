"""Fused log-softmax + NLL loss. Lifted from Chapter 20.

Computes per-token loss in one kernel: read the logit row, compute the
stable log-sum-exp, gather the target logit, write the scalar loss.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stride_lm,
    V,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    if target == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < V
    base = logits_ptr + row * stride_lm
    x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    lse = m + tl.log(tl.sum(e, axis=0))

    target_logit = tl.load(base + target).to(tl.float32)
    loss = lse - target_logit
    tl.store(loss_ptr + row, loss)


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Token-level cross-entropy loss.

    Args:
        logits: (N, V) tensor of unnormalized scores.
        targets: (N,) int64 tensor of class indices.
        ignore_index: target value that contributes zero loss.
        reduction: "mean", "sum", or "none".
    """
    if logits.dim() != 2:
        raise ValueError("cross_entropy: logits must be 2D")
    N, V = logits.shape
    losses = torch.empty(N, device=logits.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(V)
    _cross_entropy_kernel[(N,)](
        logits, targets, losses,
        logits.stride(0),
        V, ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 4096 else 4,
    )
    if reduction == "mean":
        valid = (targets != ignore_index).sum().clamp_min(1)
        return losses.sum() / valid
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"cross_entropy: unknown reduction {reduction}")
