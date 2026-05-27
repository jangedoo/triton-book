"""Tests for Chapter 23 — dequant and W8A16 matmul."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda only"
)


def _ref_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (scale[:, None].to(torch.float32) * q.to(torch.float32)).to(torch.float16)


@pytest.mark.parametrize("M,N", [(7, 11), (64, 128), (130, 257), (256, 128)])
def test_dequant_int8_per_channel(M, N):
    from src.ch23_quant import dequant_int8_per_channel

    torch.manual_seed(0)
    q = torch.randint(-128, 128, (M, N), dtype=torch.int8, device="cuda")
    scale = (torch.rand(M, device="cuda", dtype=torch.float16) * 0.05 + 0.01)

    out = dequant_int8_per_channel(q, scale)
    ref = _ref_dequant(q, scale)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
def test_dequant_dtype_variants(scale_dtype):
    from src.ch23_quant import dequant_int8_per_channel

    M, N = 32, 64
    q = torch.randint(-128, 128, (M, N), dtype=torch.int8, device="cuda")
    scale = torch.rand(M, device="cuda", dtype=scale_dtype) * 0.05 + 0.01
    out = dequant_int8_per_channel(q, scale)
    ref = _ref_dequant(q, scale.to(torch.float32))
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def _ref_gelu(x):
    k0 = math.sqrt(2.0 / math.pi)
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


@pytest.mark.parametrize("M,N", [(7, 11), (64, 128), (130, 257)])
def test_dequant_bias_gelu_fused(M, N):
    from src.ch23_quant import dequant_bias_gelu_fused

    torch.manual_seed(0)
    q = torch.randint(-128, 128, (M, N), dtype=torch.int8, device="cuda")
    scale = torch.rand(M, device="cuda", dtype=torch.float16) * 0.02 + 0.005
    bias = torch.randn(N, device="cuda", dtype=torch.float16) * 0.1

    out = dequant_bias_gelu_fused(q, scale, bias)
    x32 = scale.to(torch.float32)[:, None] * q.to(torch.float32) + bias.to(torch.float32)[None, :]
    ref = _ref_gelu(x32).to(torch.float16)
    # GELU + dequant noise: loosen atol a touch.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("M,N,K", [
    (32, 64, 32),       # one block in every dim
    (64, 128, 64),
    (65, 130, 33),      # non-pow2 / non-multiple
    (128, 256, 128),
])
def test_w8a16_matmul(M, N, K):
    from src.ch23_quant import w8a16_matmul

    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    scale = (torch.rand(N, device="cuda", dtype=torch.float16) * 0.02 + 0.005)

    y = w8a16_matmul(x, w, scale)

    # Reference: dequantize W to fp16 then standard matmul.
    w_fp = (scale[None, :].to(torch.float32) * w.to(torch.float32)).to(torch.float16)
    ref = (x.to(torch.float32) @ w_fp.to(torch.float32)).to(torch.float16)

    # fp16 matmul at K=128 needs a generous tolerance.
    torch.testing.assert_close(y, ref, rtol=3e-2, atol=3e-2)
