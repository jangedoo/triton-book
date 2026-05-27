"""Smoke tests for the consolidated `mini_triton_llm` package.

Each test imports a function via the public API and compares against a
PyTorch reference. The point is not to re-test every kernel exhaustively
(the chapter-specific test modules do that); it is to prove that the
package's public surface works end-to-end after import.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Make `mini_triton_llm` importable without installing the package.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "ch28_mini_lib"),
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


def test_public_api_imports():
    import mini_triton_llm as mtl

    for name in [
        "rmsnorm",
        "residual_rmsnorm",
        "softmax",
        "rope",
        "swiglu",
        "flash_attention",
        "cross_entropy",
        "benchmarking",
        "testing",
    ]:
        assert hasattr(mtl, name), f"missing public symbol: {name}"
    assert mtl.__version__ == "0.1.0"


def test_rmsnorm_correctness():
    from mini_triton_llm import rmsnorm
    from mini_triton_llm.testing import assert_close_fp16

    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.float16, device="cuda")
    w = torch.randn(256, dtype=torch.float16, device="cuda")
    y = rmsnorm(x, w, eps=1e-6)

    rms = torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
    ref = (x.float() * rms * w.float()).to(torch.float16)
    assert_close_fp16(y, ref)


def test_residual_rmsnorm_correctness():
    from mini_triton_llm import residual_rmsnorm
    from mini_triton_llm.testing import assert_close_fp16

    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.float16, device="cuda")
    r = torch.randn(8, 256, dtype=torch.float16, device="cuda")
    w = torch.randn(256, dtype=torch.float16, device="cuda")
    h, y = residual_rmsnorm(x, r, w)

    h_ref = (x + r).float()
    rms = torch.rsqrt((h_ref ** 2).mean(-1, keepdim=True) + 1e-6)
    y_ref = (h_ref * rms * w.float()).to(torch.float16)
    assert_close_fp16(h, h_ref.to(torch.float16))
    assert_close_fp16(y, y_ref)


def test_softmax_correctness():
    from mini_triton_llm import softmax
    from mini_triton_llm.testing import assert_close_fp16

    torch.manual_seed(0)
    x = torch.randn(16, 511, dtype=torch.float16, device="cuda")  # non-pow2
    y = softmax(x)
    ref = torch.softmax(x.float(), dim=-1).to(torch.float16)
    assert_close_fp16(y, ref)


def test_rope_correctness():
    from mini_triton_llm import rope

    torch.manual_seed(0)
    B, S, H, D = 2, 16, 4, 32
    x = torch.randn(B, S, H, D, dtype=torch.float32, device="cuda")
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    freqs = pos[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()
    y = rope(x, cos, sin, offset=0)

    x0, x1 = x[..., : D // 2], x[..., D // 2 :]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    ref = torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_swiglu_correctness():
    from mini_triton_llm import swiglu
    from mini_triton_llm.testing import assert_close_fp16

    torch.manual_seed(0)
    a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
    b = torch.randn(64, 128, dtype=torch.float16, device="cuda")
    y = swiglu(a, b)
    ref = (F.silu(a.float()) * b.float()).to(torch.float16)
    assert_close_fp16(y, ref)


def test_flash_attention_correctness():
    from mini_triton_llm import flash_attention
    from mini_triton_llm.testing import assert_close_fp16, random_qkv

    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    q, k, v = random_qkv(B, H, S, D, dtype=torch.float16)
    out = flash_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert_close_fp16(out, ref)


def test_cross_entropy_correctness():
    from mini_triton_llm import cross_entropy

    torch.manual_seed(0)
    N, V = 64, 2048
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    loss = cross_entropy(logits, targets, reduction="mean")
    ref = F.cross_entropy(logits, targets, reduction="mean")
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)
