"""Tests for Chapter 17: KV cache kernels."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")

from src.ch17_kv_cache.append_kv_cache import append_kv_cache
from src.ch17_kv_cache.decode_attention import decode_attention


# ---------------------------------------------------------------------------
# append_kv_cache
# ---------------------------------------------------------------------------


def test_append_writes_single_slot():
    B, H, max_seq, D = 1, 2, 8, 16
    k_cache = torch.zeros(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    v_new = torch.randn_like(k_new)
    append_kv_cache(k_new, v_new, k_cache, v_cache, position=3)
    torch.testing.assert_close(k_cache[:, :, 3:4, :], k_new, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_cache[:, :, 3:4, :], v_new, rtol=1e-5, atol=1e-5)
    # Other slots untouched
    untouched = torch.zeros(B, H, max_seq - 1, D, device="cuda", dtype=torch.float32)
    other = torch.cat([k_cache[:, :, :3, :], k_cache[:, :, 4:, :]], dim=2)
    torch.testing.assert_close(other, untouched, rtol=1e-5, atol=1e-5)


def test_append_roundtrip_many_tokens():
    torch.manual_seed(0)
    B, H, max_seq, D = 2, 4, 64, 32
    k_cache = torch.zeros(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    n = 17
    k_hist = torch.randn(B, H, n, D, device="cuda", dtype=torch.float32)
    v_hist = torch.randn_like(k_hist)
    for t in range(n):
        append_kv_cache(
            k_hist[:, :, t : t + 1, :].contiguous(),
            v_hist[:, :, t : t + 1, :].contiguous(),
            k_cache, v_cache, t,
        )
    torch.testing.assert_close(k_cache[:, :, :n, :], k_hist, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_cache[:, :, :n, :], v_hist, rtol=1e-5, atol=1e-5)


def test_append_non_power_of_two_dim():
    B, H, max_seq, D = 1, 2, 8, 47
    k_cache = torch.zeros(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    v_new = torch.randn_like(k_new)
    append_kv_cache(k_new, v_new, k_cache, v_cache, position=2)
    torch.testing.assert_close(k_cache[:, :, 2:3, :], k_new, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_cache[:, :, 2:3, :], v_new, rtol=1e-5, atol=1e-5)


def test_append_fp16():
    B, H, max_seq, D = 1, 2, 16, 64
    k_cache = torch.zeros(B, H, max_seq, D, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_new = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    append_kv_cache(k_new, v_new, k_cache, v_cache, position=5)
    torch.testing.assert_close(k_cache[:, :, 5:6, :], k_new, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_cache[:, :, 5:6, :], v_new, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# decode_attention
# ---------------------------------------------------------------------------


def test_decode_attention_small_matches_sdpa():
    torch.manual_seed(0)
    B, H, max_seq, D, n = 1, 2, 32, 16, 7
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    k_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)

    out = decode_attention(q, k_cache, v_cache, current_len=n)
    ref = F.scaled_dot_product_attention(q, k_cache[:, :, :n, :], v_cache[:, :, :n, :])
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_decode_attention_medium_matches_sdpa():
    torch.manual_seed(1)
    B, H, max_seq, D, n = 2, 8, 1024, 64, 513
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    k_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)

    out = decode_attention(q, k_cache, v_cache, current_len=n)
    ref = F.scaled_dot_product_attention(q, k_cache[:, :, :n, :], v_cache[:, :, :n, :])
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_decode_attention_n_equals_block():
    torch.manual_seed(2)
    B, H, max_seq, D = 1, 2, 256, 32
    for n in (1, 63, 64, 65):
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
        k_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)
        v_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)
        out = decode_attention(q, k_cache, v_cache, current_len=n)
        ref = F.scaled_dot_product_attention(q, k_cache[:, :, :n, :], v_cache[:, :, :n, :])
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_decode_attention_fp16():
    torch.manual_seed(3)
    B, H, max_seq, D, n = 1, 4, 512, 64, 257
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float16)

    out = decode_attention(q, k_cache, v_cache, current_len=n)
    ref = F.scaled_dot_product_attention(q, k_cache[:, :, :n, :], v_cache[:, :, :n, :])
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_decode_attention_ignores_stale_cache_beyond_current_len():
    """The contract says we only read the first current_len rows. Stale data past
    that point must not affect the output."""
    torch.manual_seed(4)
    B, H, max_seq, D, n = 1, 2, 128, 32, 10
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    k_cache_a = torch.randn(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache_a = torch.randn_like(k_cache_a)
    k_cache_b = k_cache_a.clone()
    v_cache_b = v_cache_a.clone()
    # Corrupt the stale region of cache_b
    k_cache_b[:, :, n:, :] = 1e3
    v_cache_b[:, :, n:, :] = -1e3

    out_a = decode_attention(q, k_cache_a, v_cache_a, current_len=n)
    out_b = decode_attention(q, k_cache_b, v_cache_b, current_len=n)
    torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Roundtrip: append then attend
# ---------------------------------------------------------------------------


def test_append_then_attend_roundtrip():
    torch.manual_seed(5)
    B, H, max_seq, D, n = 1, 4, 128, 32, 23
    k_cache = torch.zeros(B, H, max_seq, D, device="cuda", dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_hist = torch.randn(B, H, n, D, device="cuda", dtype=torch.float32)
    v_hist = torch.randn_like(k_hist)
    for t in range(n):
        append_kv_cache(
            k_hist[:, :, t : t + 1, :].contiguous(),
            v_hist[:, :, t : t + 1, :].contiguous(),
            k_cache, v_cache, t,
        )
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    out = decode_attention(q, k_cache, v_cache, current_len=n)
    ref = F.scaled_dot_product_attention(q, k_hist, v_hist)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
