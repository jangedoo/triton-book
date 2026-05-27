"""Solutions for Chapter 24 — debugging puzzles."""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: the OOB-store puzzle.
# The fix is in src/ch24_debugging/debug_demo.py:add_fixed.
# Add `mask=mask` to the tl.store call.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 2: fp16 softmax NaN.
# Missing max-subtract. Add `m = tl.max(x, axis=0); x = x - m` before exp.
# ---------------------------------------------------------------------------
@triton.jit
def fp16_softmax_fixed(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + pid * N + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)                       # FIX
    x = x - m                                   # FIX
    e = tl.exp(x.to(tl.float32))                # accumulate in fp32 too
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.float16)
    tl.store(y_ptr + pid * N + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Exercise 4: stride bug — row-wise sum that respects strides.
# ---------------------------------------------------------------------------
@triton.jit
def row_sum_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = x_ptr + pid * stride_m + offs * stride_n   # FIX: use strides
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK = triton.next_power_of_2(N)
    row_sum_kernel[(M,)](
        x, out, M, N,
        x.stride(0), x.stride(1),                       # FIX: pass strides
        BLOCK=BLOCK,
    )
    return out


# ---------------------------------------------------------------------------
# Exercise 5: dtype bug — accumulator must be fp32 even for fp16 inputs.
# ---------------------------------------------------------------------------
@triton.jit
def matmul_fixed_accumulator(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # FIX: fp32 acc
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Exercise 6: triple bug fix.
# Bugs:
#   1) Missing mask on tl.load. Add mask=(offs < N), other=0.0.
#   2) Accumulator dtype fp16. Change to fp32 and cast at store.
#   3) The original launcher used `pid * N` for row offset; if x is
#      strided (transposed), this is wrong. Use `pid * stride_m + offs * stride_n`.
# ---------------------------------------------------------------------------
@triton.jit
def fixed_sum_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N                                                  # fix 1
    x = tl.load(x_ptr + pid * stride_m + offs * stride_n,
                mask=mask, other=0.0)                                 # fix 1 + 3
    acc = tl.zeros((), dtype=tl.float32)                              # fix 2
    acc += tl.sum(x.to(tl.float32), axis=0)
    tl.store(out_ptr + pid, acc.to(x.dtype))


def fixed_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    fixed_sum_kernel[(M,)](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK=BLOCK,
    )
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    x = torch.randn(8, 100, device="cuda")
    print("row_sum vs torch:", (row_sum(x) - x.sum(dim=1)).abs().max().item())
    print("row_sum on transposed view:",
          (row_sum(x.t().contiguous().t()) - x.sum(dim=1)).abs().max().item())
