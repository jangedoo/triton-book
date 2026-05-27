"""Correctness tests for the Chapter 2 vector_add kernel."""

from __future__ import annotations

import pytest
import torch

from src.ch02_mental_model import vector_add


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@pytest.mark.parametrize("n", [7, 1024, 1023, 1025, 1_048_576])
def test_vector_add_fp32(n: int) -> None:
    """fp32 across small / medium / non-power-of-two / boundary / large shapes."""
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = vector_add(x, y)
    torch.testing.assert_close(out, _ref(x, y), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n", [1024, 4096, 99_999])
def test_vector_add_fp16(n: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.randn(n, device="cuda", dtype=torch.float16)
    out = vector_add(x, y)
    torch.testing.assert_close(out, _ref(x, y), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n", [1024, 4096, 99_999])
def test_vector_add_bf16(n: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = vector_add(x, y)
    torch.testing.assert_close(out, _ref(x, y), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("block_size", [64, 128, 256, 1024, 4096])
def test_vector_add_block_sizes(block_size: int) -> None:
    """Same output regardless of which BLOCK_SIZE the launcher uses."""
    torch.manual_seed(0)
    n = 12_345  # deliberately non-multiple of any of the block sizes
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = vector_add(x, y, BLOCK_SIZE=block_size)
    torch.testing.assert_close(out, _ref(x, y), rtol=1e-5, atol=1e-5)


def test_vector_add_zero_length() -> None:
    """Empty input is a valid edge case — grid is 0, no kernel launches."""
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    y = torch.empty(0, device="cuda", dtype=torch.float32)
    out = vector_add(x, y)
    assert out.shape == x.shape
