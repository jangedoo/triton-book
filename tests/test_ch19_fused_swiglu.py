import pytest
import torch
import torch.nn.functional as F

from ch19_fused_swiglu import swiglu_bias, swiglu_bias_ref, geglu_bias, geglu_bias_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (7, 13),          # tiny, hand-traceable
        (16, 128),        # power-of-two boundary
        (17, 129),        # non-power-of-two -> mask path
        (32, 4096),       # block boundary in H
        (33, 4097),       # boundary + 1 both axes
        (64, 11008),      # LLaMA-7B intermediate
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_against_ref(shape, dtype):
    torch.manual_seed(0)
    M, H = shape
    xg = torch.randn(M, H, device="cuda", dtype=dtype)
    bg = torch.randn(H,    device="cuda", dtype=dtype)
    xu = torch.randn(M, H, device="cuda", dtype=dtype)
    bu = torch.randn(H,    device="cuda", dtype=dtype)
    y_ref = swiglu_bias_ref(xg, bg, xu, bu)
    y = swiglu_bias(xg, bg, xu, bu)
    torch.testing.assert_close(y, y_ref, **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_no_bias(dtype):
    M, H = 32, 4096
    xg = torch.randn(M, H, device="cuda", dtype=dtype)
    xu = torch.randn(M, H, device="cuda", dtype=dtype)
    y_ref = F.silu(xg.float()).to(dtype) * xu
    y = swiglu_bias(xg, None, xu, None)
    torch.testing.assert_close(y, y_ref, **_tol(dtype))


def test_swiglu_partial_bias():
    M, H = 32, 4096
    dtype = torch.float16
    xg = torch.randn(M, H, device="cuda", dtype=dtype)
    bg = torch.randn(H,    device="cuda", dtype=dtype)
    xu = torch.randn(M, H, device="cuda", dtype=dtype)
    y_ref = F.silu((xg + bg).float()).to(dtype) * xu
    y = swiglu_bias(xg, bg, xu, None)
    torch.testing.assert_close(y, y_ref, **_tol(dtype))


@pytest.mark.parametrize("shape", [(16, 128), (33, 4097), (64, 11008)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_geglu_against_ref(shape, dtype):
    torch.manual_seed(0)
    M, H = shape
    xg = torch.randn(M, H, device="cuda", dtype=dtype)
    bg = torch.randn(H,    device="cuda", dtype=dtype)
    xu = torch.randn(M, H, device="cuda", dtype=dtype)
    bu = torch.randn(H,    device="cuda", dtype=dtype)
    y_ref = geglu_bias_ref(xg, bg, xu, bu)
    y = geglu_bias(xg, bg, xu, bu)
    torch.testing.assert_close(y, y_ref, **_tol(dtype))
