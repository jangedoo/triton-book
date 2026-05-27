import pytest
import torch
import torch.nn.functional as F

from ch20_cross_entropy import (
    cross_entropy_forward,
    cross_entropy_backward,
    cross_entropy_ref,
    logsumexp,
    logsumexp_ref,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-4)
    return dict(rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("V", [7, 128, 129, 1024, 32000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logsumexp(V, dtype):
    N = 16
    torch.manual_seed(0)
    x = torch.randn(N, V, device="cuda", dtype=dtype)
    out = logsumexp(x)
    ref = logsumexp_ref(x)
    torch.testing.assert_close(out, ref, **_tol(torch.float32))


@pytest.mark.parametrize("V", [7, 1024, 1025, 32000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cross_entropy_no_ignore(V, dtype):
    N = 32
    torch.manual_seed(0)
    logits = torch.randn(N, V, device="cuda", dtype=dtype)
    target = torch.randint(0, V, (N,), device="cuda")
    loss = cross_entropy_forward(logits, target, ignore_index=-100)
    ref = cross_entropy_ref(logits, target, ignore_index=-100)
    torch.testing.assert_close(loss, ref, **_tol(dtype))


@pytest.mark.parametrize("V", [1024, 32000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cross_entropy_with_ignore(V, dtype):
    N = 64
    torch.manual_seed(0)
    logits = torch.randn(N, V, device="cuda", dtype=dtype)
    target = torch.randint(0, V, (N,), device="cuda")
    target[::4] = -100  # 25% ignored
    loss = cross_entropy_forward(logits, target, ignore_index=-100)
    ref = cross_entropy_ref(logits, target, ignore_index=-100)
    torch.testing.assert_close(loss, ref, **_tol(dtype))


def test_cross_entropy_sum_reduction():
    N, V = 32, 1024
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (N,), device="cuda")
    loss = cross_entropy_forward(logits, target, reduction="sum")
    ref = cross_entropy_ref(logits, target, reduction="sum")
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)


def test_cross_entropy_backward():
    N, V = 32, 1024
    torch.manual_seed(0)
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, V, (N,), device="cuda")
    # Reference grad
    ref = F.cross_entropy(logits, target)
    ref.backward()
    ref_grad = logits.grad.clone()
    logits.grad = None

    _, lse = cross_entropy_forward(logits.detach(), target, return_lse=True)
    grad = cross_entropy_backward(logits.detach(), target, lse)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-4)


def test_cross_entropy_all_ignored():
    N, V = 16, 1024
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    target = torch.full((N,), -100, device="cuda")
    loss = cross_entropy_forward(logits, target)
    # clamp_min(1) protects against div by zero
    assert loss.item() == 0.0
