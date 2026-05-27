"""Small test helpers reused across the kernels."""

from __future__ import annotations

import torch


def assert_close_fp16(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`torch.testing.assert_close` with our fp16 tolerances."""
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def assert_close_fp32(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`torch.testing.assert_close` with our fp32 tolerances."""
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def random_qkv(
    B: int,
    H: int,
    S: int,
    D: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random (Q, K, V) triplet of shape (B, H, S, D)."""
    q = torch.randn(B, H, S, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, S, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, S, D, dtype=dtype, device=device) * 0.1
    return q, k, v
