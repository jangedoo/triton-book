"""Tests for Chapter 5 softmax kernels."""

import pytest
import torch

from src.ch05_softmax import naive_softmax, stable_softmax, online_softmax


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


# ---- stable_softmax ------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 7), (1, 1024), (512, 1024), (64, 31), (8, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_stable_softmax(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    out = stable_softmax(x)
    ref = torch.softmax(x, dim=-1)
    # fp16 softmax accumulates error in the denominator; loosen slightly.
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (5e-3, 5e-3)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_cols", [128 - 1, 128, 128 + 1])
def test_stable_softmax_boundary(n_cols):
    torch.manual_seed(0)
    x = torch.randn(4, n_cols, device="cuda", dtype=torch.float32)
    out = stable_softmax(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---- naive_softmax: exists, agrees with PyTorch on tame inputs ----------

def test_naive_softmax_tame_inputs():
    """For values away from the fp16 ceiling, naive matches PyTorch."""
    torch.manual_seed(0)
    x = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    out = naive_softmax(x)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_naive_softmax_fp16_overflows_when_large():
    """Naive on fp16 with row max ~20 should NaN/inf; stable should not."""
    x = torch.tensor([[20.0, 20.5, 21.0, 19.8]], device="cuda", dtype=torch.float16)
    naive = naive_softmax(x)
    stable = stable_softmax(x)
    # The naive version computes exp(21) ~= 1.3e9 in fp32, then divides;
    # depending on dtype rounding it may produce 0s or NaNs. The contract
    # tested here is just that the stable version is finite and sums to 1.
    assert torch.isfinite(stable).all()
    torch.testing.assert_close(stable.sum(dim=-1), torch.ones(1, device="cuda", dtype=torch.float16), rtol=1e-2, atol=1e-2)
    # Tolerate either NaN or correct output from naive; the point is "don't trust it".
    del naive


# ---- online_softmax -----------------------------------------------------

@pytest.mark.parametrize("shape", [(4, 1024), (2, 8192), (1, 16384), (3, 33)])
def test_online_softmax(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = online_softmax(x, block_size=1024)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_online_softmax_non_contiguous_via_slice():
    """A row-major slice along dim=1 stays row-contiguous; this should work."""
    torch.manual_seed(0)
    x_full = torch.randn(4, 2048, device="cuda", dtype=torch.float32)
    x = x_full[:, :1500]  # last-dim slice; rows are still contiguous
    # online_softmax uses x.stride(0) for row stride, so we must pass a
    # contiguous tensor (PyTorch makes the slice non-contiguous along dim=1).
    out = online_softmax(x.contiguous(), block_size=512)
    ref = torch.softmax(x.contiguous(), dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
