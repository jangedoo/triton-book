"""Correctness tests for the Chapter 4 reduction kernels."""

from __future__ import annotations

import pytest
import torch

from src.ch04_reductions import row_sum, row_max, row_mean, row_variance


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


SHAPES = [(1, 7), (4, 1024), (8, 1025), (32, 1023), (16, 4097), (128, 8192)]


# ---------------------------------------------------------------------------
# row_sum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_row_sum_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = row_sum(x)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_row_sum_low_precision(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 4096, device="cuda", dtype=dtype)
    out = row_sum(x)
    # The kernel returns fp32; the reference must also be promoted.
    ref = x.to(torch.float32).sum(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# row_max
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_row_max_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = row_max(x)
    torch.testing.assert_close(out, x.amax(dim=1), rtol=1e-5, atol=1e-5)


def test_row_max_mask_other_value() -> None:
    """Regression: with a small last tile the masked lanes must not pollute the max."""
    torch.manual_seed(0)
    # N just past a BLOCK_SIZE boundary, with all values negative — if masked
    # lanes used other=0 we would erroneously report 0 as the max.
    x = -torch.rand(4, 1025, device="cuda", dtype=torch.float32) - 1.0
    out = row_max(x)
    torch.testing.assert_close(out, x.amax(dim=1), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# row_mean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_row_mean_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = row_mean(x)
    torch.testing.assert_close(out, x.mean(dim=1), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# row_variance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_row_variance_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = row_variance(x)
    # PyTorch's default var() uses Bessel's correction (unbiased=True). The
    # kernel computes population variance, so pass unbiased=False for parity.
    ref = x.var(dim=1, unbiased=False)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_row_variance_zero_mean_shift() -> None:
    """The two-pass kernel must give a non-negative variance even with large offsets.

    The naive E[X^2] - E[X]^2 formula would round to a negative value here;
    the two-pass deviation form does not.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float32) + 1e6
    out = row_variance(x)
    assert torch.all(out >= 0)
    torch.testing.assert_close(out, x.var(dim=1, unbiased=False), rtol=1e-3, atol=1e-3)
