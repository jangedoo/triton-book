"""Tests for Chapter 26 — autotuned kernels.

These only check that the autotuned kernels match a PyTorch reference. The
benchmark file demonstrates that the autotuner picks different configs for
different shapes.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda only"
)


@pytest.mark.parametrize("M,N", [(7, 11), (32, 256), (1024, 1024), (1024, 4096)])
def test_autotuned_softmax(M, N):
    from src.ch26_autotune import autotuned_softmax
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = autotuned_softmax(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),
    (128, 256, 64),
    (513, 257, 129),       # non-multiples
    (1024, 1024, 1024),
])
def test_autotuned_matmul(M, N, K):
    from src.ch26_autotune import autotuned_matmul
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.5
    c = autotuned_matmul(a, b)
    ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)
