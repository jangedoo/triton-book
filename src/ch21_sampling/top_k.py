"""Top-k and top-p (nucleus) masking. See Chapter 21.

top_k_mask is a hybrid: torch.topk computes the per-row threshold, then a
Triton kernel does the elementwise mask in one pass.

top_p_mask is pure-PyTorch (torch.sort + cumsum + scatter). The sort dominates
runtime; writing a Triton sort is not justified for typical vocab sizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_mask_kernel(
    x_ptr, threshold_ptr, y_ptr,
    N, V, stride_x_row, stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < V
    x = tl.load(x_ptr + row * stride_x_row + cols,
                mask=mask, other=-float("inf"))
    threshold = tl.load(threshold_ptr + row)
    keep = x >= threshold
    y = tl.where(keep, x, -float("inf"))
    tl.store(y_ptr + row * stride_y_row + cols, y, mask=mask)


def top_k_mask(logits, k):
    """Set every logit below the per-row k-th largest value to -inf."""
    assert logits.is_cuda
    N, V = logits.shape
    if k >= V:
        return logits.clone()
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[:, -1].contiguous().to(logits.dtype)
    y = torch.empty_like(logits)
    BS = triton.next_power_of_2(V)
    _topk_mask_kernel[(N,)](
        logits, threshold, y,
        N, V, logits.stride(0), y.stride(0),
        BLOCK_SIZE=BS,
    )
    return y


def top_p_mask(logits, p):
    """Hybrid top-p: torch.sort + masking + unsort.

    Keeps the smallest prefix of sorted logits whose cumulative softmax mass
    is at least p. Everything else is masked to -inf in the original layout.
    """
    assert logits.is_cuda
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cum = probs.cumsum(dim=-1)
    drop = (cum - probs) > p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out
