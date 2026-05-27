"""Tests for Chapter 6 LayerNorm."""

import pytest
import torch
import torch.nn.functional as F

from src.ch06_layernorm import layernorm, layernorm_backward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


@pytest.mark.parametrize("shape", [(3, 8), (1, 1024), (128, 4096), (32, 31), (16, 768)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_layernorm_forward(shape, dtype):
    torch.manual_seed(0)
    M, H = shape
    x = torch.randn(M, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype) * 0.1 + 1.0
    b = torch.randn(H, device="cuda", dtype=dtype) * 0.01
    out = layernorm(x, w, b, eps=1e-5)
    ref = F.layer_norm(x, (H,), w, b, eps=1e-5)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("H", [128 - 1, 128, 128 + 1])
def test_layernorm_boundary(H):
    torch.manual_seed(0)
    x = torch.randn(4, H, device="cuda", dtype=torch.float32)
    w = torch.ones(H, device="cuda")
    b = torch.zeros(H, device="cuda")
    out = layernorm(x, w, b, eps=1e-5)
    ref = F.layer_norm(x, (H,), w, b, eps=1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_layernorm_higher_rank():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 128, device="cuda", dtype=torch.float32)
    w = torch.ones(128, device="cuda")
    b = torch.zeros(128, device="cuda")
    out = layernorm(x, w, b, eps=1e-5)
    ref = F.layer_norm(x, (128,), w, b, eps=1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_layernorm_backward():
    torch.manual_seed(0)
    M, H = 64, 256
    x = torch.randn(M, H, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(H, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(H, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn(M, H, device="cuda", dtype=torch.float32)

    # Reference via PyTorch autograd.
    y_ref = F.layer_norm(x, (H,), w, b, eps=1e-5)
    y_ref.backward(dy)
    dx_ref, dw_ref, db_ref = x.grad.clone(), w.grad.clone(), b.grad.clone()
    x.grad = None; w.grad = None; b.grad = None

    # Recompute mean/rstd to feed the backward.
    with torch.no_grad():
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
    dx, dw, db = layernorm_backward(dy, x.detach(), w.detach(), mean, rstd)

    torch.testing.assert_close(dx, dx_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, dw_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(db, db_ref, rtol=1e-3, atol=1e-3)
