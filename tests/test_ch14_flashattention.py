"""Correctness tests for the Chapter 14 FlashAttention forward."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")

from src.ch14_flashattention import flash_attn_forward


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("S", [64, 128])
def test_flash_matches_sdpa_small(S, causal):
    torch.manual_seed(0)
    B, H, D = 1, 2, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    got = flash_attn_forward(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Tolerances are deliberately loose. fp16 SDPA reference and our fp32-acc
    # kernel produce values that agree to about 2e-2 in practice on Turing.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("D", [32, 64, 128])
def test_flash_supports_head_dim_variants(D):
    torch.manual_seed(0)
    B, H, S = 1, 2, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    got = flash_attn_forward(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_flash_non_power_of_two_seqlen():
    """S not a multiple of BLOCK_M; exercises the row-mask path."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 100, 64  # 100 = 64 + 36
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    got = flash_attn_forward(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_flash_block_size_variants():
    """Same answer for different (BLOCK_M, BLOCK_N) choices."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    a = flash_attn_forward(q, k, v, causal=True, block_m=64, block_n=64)
    b = flash_attn_forward(q, k, v, causal=True, block_m=32, block_n=64)
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
