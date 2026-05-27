"""Vector addition kernel — the "hello world" of Triton.

This module exposes:

- ``vector_add_kernel``: the Triton kernel. One program instance owns one
  contiguous block of ``BLOCK_SIZE`` elements of the output.
- ``vector_add``: a thin Python launcher that allocates the output, computes
  the 1-D grid, and calls the kernel.

The kernel is intentionally minimal. It is the reference for every "how do I
write a Triton kernel" question for the rest of the book.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,            # *Pointer* to the first element of x.
    y_ptr,            # *Pointer* to the first element of y.
    out_ptr,          # *Pointer* to the first element of out.
    n_elements,       # Total number of elements in x / y / out.
    BLOCK_SIZE: tl.constexpr,  # Compile-time tile width. Picked by the launcher.
):
    """Compute ``out = x + y`` over one tile.

    Each program instance covers ``[pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE)``
    in the output. The last program may run off the end of the tensor; we
    handle that with a boolean mask on the load and store.
    """
    # Which tile does this program own?
    pid = tl.program_id(axis=0)

    # Element offsets this program reads and writes.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask off the lanes that fall past the end of the tensor.
    mask = offsets < n_elements

    # Load, compute, store. Triton handles vectorization across the BLOCK_SIZE
    # lanes for you — there is no explicit per-lane loop here.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """Launch the vector_add kernel.

    Parameters
    ----------
    x, y : torch.Tensor
        Same shape, same dtype, both on CUDA, both contiguous.
    BLOCK_SIZE : int, optional
        Tile width passed as a ``constexpr`` to the kernel. 1024 is a fine
        default for elementwise kernels on most modern NVIDIA hardware.

    Returns
    -------
    torch.Tensor
        ``x + y``, same shape and dtype as ``x``.
    """
    assert x.is_cuda and y.is_cuda, "vector_add requires CUDA tensors"
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} vs {y.dtype}"

    out = torch.empty_like(x)
    n_elements = out.numel()

    # 1-D grid. The lambda form lets Triton recompute the grid for autotuned
    # block sizes if you add autotune later. For this chapter we keep the
    # block size fixed.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    vector_add_kernel[grid](
        x, y, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
