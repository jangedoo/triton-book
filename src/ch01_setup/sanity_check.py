"""Sanity check for the book's environment.

Run with:

    uv run python src/ch01_setup/sanity_check.py

If the script prints torch / triton versions, the device name and capability,
and finishes with ``ok``, the stack is wired up correctly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def print_environment() -> None:
    """Print the versions and device details that matter for this book."""
    print(f"torch  : {torch.__version__}")
    print(f"triton : {triton.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Reinstall PyTorch from the CUDA index URL "
            "given by https://pytorch.org/get-started/locally/."
        )
    print(f"device : {torch.cuda.get_device_name()}")
    print(f"cap    : {torch.cuda.get_device_capability()}")


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two 1-D CUDA tensors with a minimal Triton kernel."""
    assert x.is_cuda and y.is_cuda, "inputs must live on a CUDA device"
    assert x.shape == y.shape, "shape mismatch"
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def main() -> None:
    print_environment()
    # 1_000_003 is intentionally not a multiple of BLOCK_SIZE so the mask path
    # in the kernel actually runs.
    x = torch.randn(1_000_003, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    out = vector_add(x, y)
    torch.testing.assert_close(out, x + y, rtol=1e-5, atol=1e-5)
    print("ok")


if __name__ == "__main__":
    main()
