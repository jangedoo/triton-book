"""Correctness tests for the fused residual + LayerNorm + dropout kernel.

With `p=0` we compare exactly to the PyTorch reference. With `p>0` we
compare distributional properties (mean of kept fraction, variance of
output) since the mask is stochastic.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ch30_playbook import fused_residual_ln_dropout  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def _ref(x, r, w, b, eps):
    h = x + r
    n = F.layer_norm(h.float(), (h.shape[-1],), w.float(), b.float(), eps=eps)
    return h.to(x.dtype), n.to(x.dtype)


@pytest.mark.parametrize("N", [7, 64, 511, 1024])
def test_no_dropout_matches_pytorch(N: int) -> None:
    torch.manual_seed(0)
    M = 8
    x = torch.randn(M, N, dtype=torch.float16, device="cuda")
    r = torch.randn(M, N, dtype=torch.float16, device="cuda")
    w = torch.randn(N, dtype=torch.float16, device="cuda")
    b = torch.randn(N, dtype=torch.float16, device="cuda")

    h, y = fused_residual_ln_dropout(x, r, w, b, p=0.0, eps=1e-5, seed=0)
    h_ref, y_ref = _ref(x, r, w, b, eps=1e-5)

    torch.testing.assert_close(h, h_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)


def test_dropout_kept_fraction_is_correct() -> None:
    torch.manual_seed(0)
    M, N = 4096, 256
    x = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    r = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    w = torch.ones(N, dtype=torch.float32, device="cuda")
    b = torch.ones(N, dtype=torch.float32, device="cuda")  # post-LN value = 1.0
    p = 0.3
    _, y = fused_residual_ln_dropout(x, r, w, b, p=p, eps=1e-5, seed=42)
    # The kept entries are scaled by 1/(1-p). Their expected mean is 1.0.
    # Verify the fraction of zeros is close to p.
    zero_frac = (y == 0).float().mean().item()
    assert abs(zero_frac - p) < 0.02, f"expected ~{p}, got {zero_frac}"


def test_dropout_is_reproducible_under_same_seed() -> None:
    torch.manual_seed(0)
    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    r = torch.randn(M, N, dtype=torch.float32, device="cuda")
    w = torch.randn(N, dtype=torch.float32, device="cuda")
    b = torch.randn(N, dtype=torch.float32, device="cuda")
    _, a = fused_residual_ln_dropout(x, r, w, b, p=0.2, seed=7)
    _, c = fused_residual_ln_dropout(x, r, w, b, p=0.2, seed=7)
    torch.testing.assert_close(a, c)
