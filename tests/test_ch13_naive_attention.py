"""Correctness tests for the Chapter 13 naive attention pipeline."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")

from src.ch13_naive_attention import (
    naive_attention_forward,
    qkt_matmul,
    apply_causal_mask,
    pv_matmul,
)


@pytest.mark.parametrize("S", [64, 128, 127])  # power-of-2 and non-power-of-2
def test_qkt_matmul_matches_torch(S):
    torch.manual_seed(0)
    B, H, D = 1, 2, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    expected = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
    got = qkt_matmul(q, k)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_causal_mask_in_place():
    B, H, S = 1, 1, 8
    scores = torch.arange(S * S, device="cuda", dtype=torch.float32).reshape(1, 1, S, S).clone()
    expected = scores.clone()
    causal = torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), diagonal=1)
    expected = expected.masked_fill(causal, float("-inf"))

    apply_causal_mask(scores)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)


def test_pv_matmul_matches_torch():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 64, 32
    probs = torch.softmax(torch.randn(B, H, S, S, device="cuda"), dim=-1).to(torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    expected = (probs.float() @ v.float()).to(torch.float16)
    got = pv_matmul(probs, v)
    # PV is a softmax-weighted average; fp16 accumulation in V causes some drift.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("S", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_naive_attention_matches_sdpa(S, causal):
    torch.manual_seed(0)
    B, H, D = 1, 2, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    got = naive_attention_forward(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    # Two matmuls + softmax in fp16 accumulate roughly 3x the per-op error.
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
