"""Tests for Chapter 15: educational attention backward."""

import math
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")

from src.ch15_attention_backward.educational_backward import attention_backward_educational


def _ref_grads(q, k, v, do):
    q_ = q.detach().to(torch.float32).requires_grad_(True)
    k_ = k.detach().to(torch.float32).requires_grad_(True)
    v_ = v.detach().to(torch.float32).requires_grad_(True)
    o = F.scaled_dot_product_attention(q_, k_, v_)
    return torch.autograd.grad(o, [q_, k_, v_], do.to(torch.float32))


def test_small_grads_match_autograd():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 16, 16
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q); do = torch.randn_like(q)

    dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)
    dq, dk, dv = attention_backward_educational(q, k, v, do)
    torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)


def test_medium_grads_match_autograd():
    torch.manual_seed(1)
    B, H, S, D = 2, 4, 128, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q); do = torch.randn_like(q)

    dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)
    dq, dk, dv = attention_backward_educational(q, k, v, do)
    torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)


def test_non_power_of_two_seq():
    torch.manual_seed(2)
    B, H, S, D = 1, 2, 47, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q); do = torch.randn_like(q)

    dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)
    dq, dk, dv = attention_backward_educational(q, k, v, do)
    torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)


def test_boundary_seq_equals_block():
    # BLOCK_M and BLOCK_N are both 32 in the launcher; exercise S = 32 and S = 33.
    for S in (32, 33):
        B, H, D = 1, 2, 32
        torch.manual_seed(3)
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn_like(q); v = torch.randn_like(q); do = torch.randn_like(q)

        dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)
        dq, dk, dv = attention_backward_educational(q, k, v, do)
        torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)


def test_fp16_inputs_upcast_internally():
    torch.manual_seed(4)
    B, H, S, D = 1, 2, 64, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q); do = torch.randn_like(q)

    dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do)
    dq, dk, dv = attention_backward_educational(q, k, v, do)
    # fp16 inputs imply ~fp16 accuracy on the reference side.
    torch.testing.assert_close(dq, dq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dk, dk_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=1e-2, atol=1e-2)
