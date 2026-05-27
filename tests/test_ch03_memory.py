"""Correctness tests for the Chapter 3 memory kernels."""

from __future__ import annotations

import pytest
import torch

from src.ch03_memory import copy, row_add, transpose


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [7, 1024, 1023, 1025, 1_048_576])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_copy_exact(n: int, dtype: torch.dtype) -> None:
    """Copy is bit-exact, no arithmetic to round."""
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = copy(x)
    # Pure copy: equality, not assert_close.
    assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# row_add
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 7), (4, 1024), (8, 1025), (32, 1023), (512, 4096)])
def test_row_add_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    out = row_add(x, bias)
    torch.testing.assert_close(out, x + bias, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(4, 1024), (32, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_row_add_low_precision(shape: tuple[int, int], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)
    out = row_add(x, bias)
    torch.testing.assert_close(out, x + bias, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("block_size_n", [64, 128, 1024, 4096])
def test_row_add_block_sizes(block_size_n: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(33, 8191, device="cuda", dtype=torch.float32)
    bias = torch.randn(8191, device="cuda", dtype=torch.float32)
    out = row_add(x, bias, BLOCK_SIZE_N=block_size_n)
    torch.testing.assert_close(out, x + bias, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(7, 5), (32, 32), (33, 31), (128, 256), (1024, 1024)])
def test_transpose_fp32(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = transpose(x)
    assert torch.equal(out, x.t().contiguous())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_transpose_dtypes(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=dtype)
    out = transpose(x)
    assert torch.equal(out, x.t().contiguous())
