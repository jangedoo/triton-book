"""Solutions for Chapter 23 exercises.

Run any single exercise via its function. These are written for clarity, not
peak performance — the exercises let you read the kernel structure end to end.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: per-tensor dequant.
# ---------------------------------------------------------------------------
@triton.jit
def _per_tensor_dequant_kernel(
    q_ptr, out_ptr, scale, n_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    q = tl.load(q_ptr + offs, mask=mask, other=0)
    out = (q.to(tl.float32) * scale).to(tl.float16)
    tl.store(out_ptr + offs, out, mask=mask)


def per_tensor_dequant(q: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float16)
    n = q.numel()
    grid = (triton.cdiv(n, 1024),)
    _per_tensor_dequant_kernel[grid](q, out, float(scale), n, BLOCK=1024)
    return out


# ---------------------------------------------------------------------------
# Exercise 2: per-channel dequant with axis=1 (scale shape (N,)).
# ---------------------------------------------------------------------------
@triton.jit
def _per_channel_axis1_kernel(
    q_ptr, scale_ptr, out_ptr,
    M, N,
    stride_qm, stride_qn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn,
        mask=mask_m[:, None] & mask_n[None, :], other=0,
    )
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = (s[None, :] * q.to(tl.float32)).to(tl.float16)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out, mask=mask_m[:, None] & mask_n[None, :],
    )


def per_channel_dequant_axis1(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, N = q.shape
    out = torch.empty((M, N), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _per_channel_axis1_kernel[grid](
        q, scale, out, M, N,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=128,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 3: fused dequant + bias (no activation).
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_bias_kernel(
    q_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N,
    stride_qm, stride_qn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn,
        mask=mask_m[:, None] & mask_n[None, :], other=0,
    )
    s = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = (s[:, None] * q.to(tl.float32) + b[None, :]).to(tl.float16)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out, mask=mask_m[:, None] & mask_n[None, :],
    )


def dequant_bias(q, scale, bias):
    M, N = q.shape
    out = torch.empty((M, N), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _dequant_bias_kernel[grid](
        q, scale, bias, out, M, N,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=128,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 4: W8A16 with weight stored (N, K) row-major and per-row scale.
# ---------------------------------------------------------------------------
# Prose: this is the layout most production kernels use. The weight is
# physically (N, K) so each output channel is a contiguous row in memory,
# which makes the per-channel scale a per-row scale. Inside the K-loop,
# you load a (BLOCK_N, BLOCK_K) tile of W, broadcast the (BLOCK_N,) scale
# along K, and tl.dot against x_tile transposed.
#
# Kernel-level changes vs the chapter version:
#     - W is (N, K) instead of (K, N).
#     - stride_wn is the row stride, stride_wk is 1.
#     - The dot becomes tl.dot(x_tile, tl.trans(w_fp)) OR you can do
#       tl.dot(w_fp, tl.trans(x_tile)) and store the transposed result.
#
# We leave the full kernel as an exercise; the chapter kernel covers the
# (K, N) layout and the diff is mechanical.


# ---------------------------------------------------------------------------
# Exercise 5: W4A16 matmul sketch — unpack two int4 per byte.
# ---------------------------------------------------------------------------
# Prose sketch — not run:
#
# @triton.jit
# def w4a16_kernel(..., BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
#     # The packed weight has BLOCK_K // 2 bytes per column tile.
#     # We load BLOCK_K // 2 bytes, unpack into BLOCK_K int4 values.
#     packed_offs_k = tl.arange(0, BLOCK_K // 2)
#     packed = tl.load(w_ptr + packed_offs_k[:, None] * stride + ...)  # int8 bytes
#     low  = (packed & 0xF)                # lower nibble
#     high = (packed >> 4) & 0xF           # upper nibble
#     # sign-extend if signed int4: values 8..15 represent -8..-1
#     low  = tl.where(low  >= 8, low  - 16, low)
#     high = tl.where(high >= 8, high - 16, high)
#     # interleave the two halves back into BLOCK_K rows of int4 values.
#     # In Triton you usually keep them as two separate tiles of shape
#     # (BLOCK_K // 2, BLOCK_N) and do two tl.dot calls, accumulating
#     # both into the same fp32 accumulator. Multiply each tile by the
#     # per-column scale before the dot, exactly like W8A16.
#
# The remaining structure (X load, fp32 accumulator, fp16 store) is
# identical to w8a16_matmul.py — only the weight unpack changes.


# ---------------------------------------------------------------------------
# Exercise 6: W8A8 matmul with int8 tensor cores (sm_75+).
# ---------------------------------------------------------------------------
@triton.jit
def _w8a8_matmul_kernel(
    x_ptr, w_ptr,
    x_scale_ptr,    # (M,) fp16, per-row activation scale
    w_scale_ptr,    # (N,) fp16, per-column weight scale
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        # int8 x int8 -> int32 on Turing+ tensor cores
        acc += tl.dot(x, w, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    xs = tl.load(x_scale_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    ws = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w8a8_matmul(x_int8, w_int8, x_scale, w_scale):
    M, K = x_int8.shape
    K2, N = w_int8.shape
    assert K == K2
    y = torch.empty((M, N), dtype=torch.float16, device=x_int8.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _w8a8_matmul_kernel[grid](
        x_int8, w_int8, x_scale, w_scale, y,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
    )
    return y


if __name__ == "__main__":
    # Quick smoke check.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    q = torch.randint(-128, 128, (256, 256), dtype=torch.int8, device="cuda")
    out = per_tensor_dequant(q, 0.01)
    print("ex1 per-tensor dequant ok:", out.shape, out.dtype)

    scale_n = torch.rand(256, device="cuda", dtype=torch.float16) * 0.02 + 0.005
    out2 = per_channel_dequant_axis1(q, scale_n)
    print("ex2 per-channel axis=1 ok:", out2.shape)
