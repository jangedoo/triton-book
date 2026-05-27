"""Tests for the Chapter 9 matmul kernels."""

from __future__ import annotations

import pytest
import torch

from src.ch09_matmul import matmul_naive, matmul_grouped


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


@pytest.mark.parametrize("kernel", [matmul_naive, matmul_grouped])
def test_small(kernel):
    torch.manual_seed(0)
    a = torch.randn((32, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 32), device="cuda", dtype=torch.float16)
    out = kernel(a, b)
    ref = _ref(a.float(), b.float()).to(torch.float16)
    # fp16 accumulation tolerance
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kernel", [matmul_naive, matmul_grouped])
def test_medium(kernel):
    torch.manual_seed(0)
    a = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    out = kernel(a, b)
    ref = _ref(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kernel", [matmul_naive, matmul_grouped])
def test_non_pow2_with_mask(kernel):
    # M, N, K are all non-power-of-two -- exercises every mask path.
    torch.manual_seed(0)
    a = torch.randn((200, 150), device="cuda", dtype=torch.float16)
    b = torch.randn((150, 300), device="cuda", dtype=torch.float16)
    out = kernel(a, b)
    ref = _ref(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_naive_fp32_accumulation_precision():
    """fp16 input + fp32 accumulation should track an fp32 reference closely."""
    torch.manual_seed(0)
    M, N, K = 256, 256, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    out = matmul_naive(a, b).to(torch.float32)
    ref = _ref(a.float(), b.float())
    # Looser than usual because of the fp16 final cast.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_boundary_block_size():
    torch.manual_seed(0)
    # K exactly equal to BLOCK_K = 32.
    a = torch.randn((128, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 128), device="cuda", dtype=torch.float16)
    out = matmul_naive(a, b)
    ref = _ref(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
