"""Tests for Chapter 16: RoPE and ALiBi."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")

from src.ch16_rope.rope import (
    build_cos_sin_cache,
    rope_noninterleaved,
    rope_interleaved,
)
from src.ch16_rope.alibi import alibi_bias, build_alibi_slopes


# ---------------------------------------------------------------------------
# Reference implementations (kept here for clarity; not imported from src)
# ---------------------------------------------------------------------------


def ref_noninterleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    S = x.shape[-2]
    D = x.shape[-1]
    x1 = x[..., : D // 2].float()
    x2 = x[...,   D // 2 :].float()
    c = cos[offset : offset + S][None, None, :, :]
    s = sin[offset : offset + S][None, None, :, :]
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., : D // 2] = x1 * c - x2 * s
    out[...,   D // 2 :] = x1 * s + x2 * c
    return out.to(x.dtype)


def ref_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    S = x.shape[-2]
    D = x.shape[-1]
    xe = x[..., 0::2].float()
    xo = x[..., 1::2].float()
    c = cos[offset : offset + S][None, None, :, :]
    s = sin[offset : offset + S][None, None, :, :]
    rot_e = xe * c - xo * s
    rot_o = xe * s + xo * c
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., 0::2] = rot_e
    out[..., 1::2] = rot_o
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# RoPE non-interleaved
# ---------------------------------------------------------------------------


def test_rope_noninterleaved_small():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 7, 8
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
    out = rope_noninterleaved(x, cos, sin)
    ref = ref_noninterleaved(x, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rope_noninterleaved_medium_fp16():
    torch.manual_seed(1)
    B, H, S, D = 2, 4, 512, 64
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
    out = rope_noninterleaved(x, cos, sin)
    ref = ref_noninterleaved(x, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_rope_noninterleaved_non_power_of_two_seq():
    torch.manual_seed(2)
    B, H, S, D = 1, 2, 113, 32
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
    out = rope_noninterleaved(x, cos, sin)
    ref = ref_noninterleaved(x, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rope_noninterleaved_block_boundary():
    # default BLOCK_S = 32; exercise S = 32 and S = 33
    for S in (32, 33):
        B, H, D = 1, 2, 32
        x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
        out = rope_noninterleaved(x, cos, sin)
        ref = ref_noninterleaved(x, cos, sin)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rope_pos_offset_decode():
    """Single token at absolute position P matches the full-prefill slice."""
    torch.manual_seed(3)
    B, H, S, D = 1, 2, 32, 32
    x_full = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")

    full_rotated = rope_noninterleaved(x_full, cos, sin, pos_offset=0)
    for P in (0, 1, 7, 31):
        single = rope_noninterleaved(x_full[:, :, P : P + 1].contiguous(), cos, sin, pos_offset=P)
        torch.testing.assert_close(single, full_rotated[:, :, P : P + 1], rtol=1e-5, atol=1e-5)


def test_rope_position_zero_is_identity():
    torch.manual_seed(4)
    B, H, D = 1, 2, 16
    x = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=1, dim=D, device="cuda")
    out = rope_noninterleaved(x, cos, sin, pos_offset=0)
    # At position 0, theta = 0 for every pair, so the rotation is the identity.
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# RoPE interleaved
# ---------------------------------------------------------------------------


def test_rope_interleaved_small():
    torch.manual_seed(5)
    B, H, S, D = 1, 2, 7, 8
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
    out = rope_interleaved(x, cos, sin)
    ref = ref_interleaved(x, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rope_interleaved_medium():
    torch.manual_seed(6)
    B, H, S, D = 2, 4, 256, 64
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")
    out = rope_interleaved(x, cos, sin)
    ref = ref_interleaved(x, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------


def test_alibi_bias_matches_reference():
    H, S = 8, 64
    bias = alibi_bias(H, S, device="cuda")
    slopes = build_alibi_slopes(H, device="cuda")
    i = torch.arange(S, device="cuda")
    rel = (i[:, None] - i[None, :]).abs().to(torch.float32)
    ref = -slopes[:, None, None] * rel[None, :, :]
    torch.testing.assert_close(bias, ref, rtol=1e-5, atol=1e-5)


def test_alibi_bias_non_power_of_two_seq():
    H, S = 4, 47
    bias = alibi_bias(H, S, device="cuda")
    slopes = build_alibi_slopes(H, device="cuda")
    i = torch.arange(S, device="cuda")
    rel = (i[:, None] - i[None, :]).abs().to(torch.float32)
    ref = -slopes[:, None, None] * rel[None, :, :]
    torch.testing.assert_close(bias, ref, rtol=1e-5, atol=1e-5)


def test_alibi_diagonal_is_zero():
    H, S = 4, 32
    bias = alibi_bias(H, S, device="cuda")
    diag = torch.diagonal(bias, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag), rtol=1e-5, atol=1e-5)
