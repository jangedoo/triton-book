"""Tests for Chapter 10 batched matmul and fused linear+bias+GELU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.ch10_batched_linear import batched_matmul, linear_bias_gelu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def test_batched_matmul_small():
    torch.manual_seed(0)
    a = torch.randn((4, 64, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((4, 32, 48), device="cuda", dtype=torch.float16)
    out = batched_matmul(a, b)
    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_batched_matmul_non_pow2():
    torch.manual_seed(0)
    a = torch.randn((3, 100, 70), device="cuda", dtype=torch.float16)
    b = torch.randn((3, 70, 130), device="cuda", dtype=torch.float16)
    out = batched_matmul(a, b)
    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_batched_matmul_large_batch():
    torch.manual_seed(0)
    a = torch.randn((16, 128, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 256, 128), device="cuda", dtype=torch.float16)
    out = batched_matmul(a, b)
    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_linear_bias_gelu_small():
    torch.manual_seed(0)
    x = torch.randn((2, 16, 32), device="cuda", dtype=torch.float16)
    w = torch.randn((48, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((48,),     device="cuda", dtype=torch.float16)
    out = linear_bias_gelu(x, w, b)
    ref = F.gelu(F.linear(x.float(), w.float(), b.float()), approximate="tanh").to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_linear_bias_gelu_3d_batch():
    torch.manual_seed(0)
    x = torch.randn((4, 128, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((512, 256),    device="cuda", dtype=torch.float16)
    b = torch.randn((512,),        device="cuda", dtype=torch.float16)
    out = linear_bias_gelu(x, w, b)
    ref = F.gelu(F.linear(x.float(), w.float(), b.float()), approximate="tanh").to(torch.float16)
    # Looser because long K means more fp16 cast error.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_linear_bias_gelu_non_pow2():
    torch.manual_seed(0)
    x = torch.randn((1, 50, 70),  device="cuda", dtype=torch.float16)
    w = torch.randn((130, 70),    device="cuda", dtype=torch.float16)
    b = torch.randn((130,),       device="cuda", dtype=torch.float16)
    out = linear_bias_gelu(x, w, b)
    ref = F.gelu(F.linear(x.float(), w.float(), b.float()), approximate="tanh").to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
