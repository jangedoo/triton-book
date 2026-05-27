"""Tests for Chapter 7 RMSNorm."""

import pytest
import torch

from src.ch07_rmsnorm import rmsnorm, rmsnorm_backward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _ref_rmsnorm(x, w, eps):
    # Compute in fp32 to match the kernel's accumulator.
    x32 = x.to(torch.float32)
    mean_sq = (x32 * x32).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(mean_sq + eps)
    y = x32 * rstd * w.to(torch.float32)
    return y.to(x.dtype)


@pytest.mark.parametrize("shape", [(3, 7), (1, 1024), (128, 768), (32, 31), (16, 4096), (4, 8192)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_forward(shape, dtype):
    torch.manual_seed(0)
    M, H = shape
    x = torch.randn(M, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype) * 0.1 + 1.0
    out = rmsnorm(x, w, eps=1e-6)
    ref = _ref_rmsnorm(x, w, 1e-6)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("H", [128 - 1, 128, 128 + 1])
def test_rmsnorm_boundary(H):
    torch.manual_seed(0)
    x = torch.randn(4, H, device="cuda", dtype=torch.float32)
    w = torch.ones(H, device="cuda")
    out = rmsnorm(x, w, eps=1e-6)
    ref = _ref_rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rmsnorm_higher_rank():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 128, device="cuda", dtype=torch.float32)
    w = torch.ones(128, device="cuda")
    out = rmsnorm(x, w, eps=1e-6)
    ref = _ref_rmsnorm(x.reshape(-1, 128), w, 1e-6).reshape(x.shape)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rmsnorm_backward():
    torch.manual_seed(0)
    M, H = 64, 256
    x = torch.randn(M, H, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(H, device="cuda", dtype=torch.float32, requires_grad=True) * 0.1 + 1.0
    dy = torch.randn(M, H, device="cuda", dtype=torch.float32)

    # PyTorch autograd reference
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rstd_ref = 1.0 / torch.sqrt(mean_sq + 1e-6)
    y_ref = x * rstd_ref * w
    y_ref.backward(dy)
    dx_ref, dw_ref = x.grad.clone(), w.grad.clone()

    with torch.no_grad():
        ms = (x * x).mean(dim=-1)
        rstd = 1.0 / torch.sqrt(ms + 1e-6)
    dx, dw = rmsnorm_backward(dy, x.detach(), w.detach(), rstd)
    torch.testing.assert_close(dx, dx_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, dw_ref, rtol=1e-3, atol=1e-3)
