import pytest
import torch

from ch21_sampling import temperature_scale, argmax_sample, top_k_mask, top_p_mask


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


@pytest.mark.parametrize("V", [7, 1024, 1025, 32000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_temperature(V, dtype):
    N = 4
    torch.manual_seed(0)
    x = torch.randn(N, V, device="cuda", dtype=dtype)
    y = temperature_scale(x, 0.7)
    ref = x / 0.7
    tol = dict(rtol=1e-2, atol=1e-2) if dtype != torch.float32 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, ref, **tol)


def test_temperature_identity():
    x = torch.randn(4, 1024, device="cuda")
    y = temperature_scale(x, 1.0)
    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("V", [7, 1024, 1025, 32000, 128000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_argmax(V, dtype):
    N = 8
    torch.manual_seed(0)
    x = torch.randn(N, V, device="cuda", dtype=dtype)
    idx = argmax_sample(x)
    ref = x.argmax(dim=-1)
    assert idx.dtype == torch.int64
    torch.testing.assert_close(idx, ref)


def test_argmax_decode_shape():
    # one row, one sampled id
    x = torch.randn(1, 50272, device="cuda")
    idx = argmax_sample(x)
    assert idx.shape == (1,)
    assert idx.dtype == torch.int64


@pytest.mark.parametrize("k", [1, 10, 50, 256])
def test_topk_mask_count(k):
    N, V = 4, 1025
    torch.manual_seed(0)
    x = torch.randn(N, V, device="cuda")
    y = top_k_mask(x, k)
    finite_per_row = (y != float("-inf")).sum(dim=-1)
    # >= k because of ties; in practice for random fp32 logits, exactly k
    assert (finite_per_row >= k).all()
    # the surviving values are exactly the top-k values of x
    survivors = torch.where(y != float("-inf"), y, torch.full_like(y, float("-inf")))
    top_kept = survivors.topk(k, dim=-1).values
    top_ref = x.topk(k, dim=-1).values
    torch.testing.assert_close(top_kept, top_ref)


def test_topk_mask_k_ge_v():
    x = torch.randn(2, 16, device="cuda")
    y = top_k_mask(x, k=16)
    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("p", [0.5, 0.9, 0.95])
def test_top_p_mask(p):
    N, V = 4, 1024
    torch.manual_seed(0)
    x = torch.randn(N, V, device="cuda")
    y = top_p_mask(x, p)
    # surviving logits' softmax mass must be >= p for every row
    surviving_mask = y != float("-inf")
    probs = torch.softmax(x.float(), dim=-1)
    kept_mass = (probs * surviving_mask).sum(dim=-1)
    assert (kept_mass >= p - 1e-4).all(), f"kept_mass={kept_mass}"
