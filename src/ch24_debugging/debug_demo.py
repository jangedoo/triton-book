"""A deliberately buggy add kernel paired with the fix.

The bug: the launcher computes a grid that overshoots the array length, and
the kernel forgets the mask on `tl.store`. With small array sizes that are
not multiples of BLOCK, the last program writes past the end of `out` and
corrupts whatever lives there in memory.

The bug only shows up when:
    n_elements is not a multiple of BLOCK
    AND the tail program runs (it always does)
    AND you look at memory you did not allocate (otherwise you may not notice)

Reproduction recipe in the chapter:
    1) Run with n_elements = 1024 (clean multiple) — passes.
    2) Run with n_elements = 1000 — silently passes most of the time, fails
       intermittently, or trashes a neighboring tensor.
    3) Enable TRITON_INTERPRET=1 and add tl.device_print on offs to see the
       OOB write.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Buggy version: missing mask on tl.store.
# ---------------------------------------------------------------------------
@triton.jit
def _add_buggy_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = x + y
    # BUG: no mask here. Writes BLOCK elements regardless of n_elements.
    tl.store(out_ptr + offs, out)


def add_with_mask_bug(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = (triton.cdiv(n, BLOCK),)
    _add_buggy_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------------------
# Fixed version: same kernel with the mask on tl.store.
# ---------------------------------------------------------------------------
@triton.jit
def _add_fixed_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offs, out, mask=mask)   # FIX


def add_fixed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = (triton.cdiv(n, BLOCK),)
    _add_fixed_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out
