import pytest
import torch

from ch18_residual_rmsnorm import residual_rmsnorm, residual_rmsnorm_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 7),         # tiny, hand-traceable
        (4, 1024),      # power of two
        (3, 1025),      # non-power-of-two -> mask path
        (1, 128),       # boundary == small block
        (5, 129),       # boundary + 1
        (8, 4096),      # realistic LLM hidden
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_against_ref(shape, dtype):
    torch.manual_seed(0)
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    r = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)

    y_ref, t_ref = residual_rmsnorm_ref(x, r, w)
    y, t = residual_rmsnorm(x, r, w)

    torch.testing.assert_close(y, y_ref, **_tol(dtype))
    torch.testing.assert_close(t, t_ref, **_tol(dtype))


def test_no_weight():
    M, N = 4, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    r = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y_ref, _ = residual_rmsnorm_ref(x, r, None)
    y, _ = residual_rmsnorm(x, r, None)
    torch.testing.assert_close(y, y_ref, **_tol(torch.float16))


def test_no_return_residual():
    M, N = 4, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    r = torch.randn(M, N, device="cuda", dtype=torch.float16)
    w = torch.randn(N, device="cuda", dtype=torch.float16)
    y_ref, _ = residual_rmsnorm_ref(x, r, w)
    y = residual_rmsnorm(x, r, w, return_residual=False)
    torch.testing.assert_close(y, y_ref, **_tol(torch.float16))


def test_three_d_input():
    B, T, H = 2, 16, 1024
    x = torch.randn(B, T, H, device="cuda", dtype=torch.float16)
    r = torch.randn(B, T, H, device="cuda", dtype=torch.float16)
    w = torch.randn(H, device="cuda", dtype=torch.float16)
    y, t = residual_rmsnorm(x, r, w)
    y_ref, t_ref = residual_rmsnorm_ref(x, r, w)
    torch.testing.assert_close(y, y_ref, **_tol(torch.float16))
    torch.testing.assert_close(t, t_ref, **_tol(torch.float16))
