"""Tests for Chapter 8 activations."""

import pytest
import torch
import torch.nn.functional as F

from src.ch08_activations import gelu, silu, swiglu, geglu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


# ---- GELU ---------------------------------------------------------------

@pytest.mark.parametrize("shape", [(7,), (1024,), (128, 128), (32, 31), (4, 1024 + 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(shape, dtype, approximate):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    out = gelu(x, approximate=approximate)
    ref = F.gelu(x, approximate=approximate)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ---- SiLU ---------------------------------------------------------------

@pytest.mark.parametrize("shape", [(7,), (1024,), (128, 128), (32, 31)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_silu(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    out = silu(x)
    ref = F.silu(x)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ---- SwiGLU -------------------------------------------------------------

@pytest.mark.parametrize("shape", [(128, 128), (16, 4096), (32, 31)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu(shape, dtype):
    torch.manual_seed(0)
    g = torch.randn(*shape, device="cuda", dtype=dtype)
    u = torch.randn(*shape, device="cuda", dtype=dtype)
    out = swiglu(g, u)
    ref = F.silu(g) * u
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ---- GEGLU --------------------------------------------------------------

@pytest.mark.parametrize("shape", [(128, 128), (16, 4096), (32, 31)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_geglu(shape, dtype):
    torch.manual_seed(0)
    g = torch.randn(*shape, device="cuda", dtype=dtype)
    u = torch.randn(*shape, device="cuda", dtype=dtype)
    out = geglu(g, u)
    ref = F.gelu(g) * u
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
