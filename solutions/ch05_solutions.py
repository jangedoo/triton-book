"""Chapter 5 worked solutions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Exercise 1: Causal-masked softmax ----------------------------------------

@triton.jit
def _causal_softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    in_bounds = col_offs < n_cols
    causal = col_offs <= row_idx
    mask = in_bounds & causal
    x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(y_ptr + row_idx * row_stride + col_offs, y.to(tl.load(row_start + col_offs, mask=in_bounds, other=0.0).dtype), mask=in_bounds)


def causal_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.zeros_like(x)
    _causal_softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


# Exercise 2: Temperature scaling ------------------------------------------

@triton.jit
def _temp_softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, tau, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols
    x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x / tau
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row_idx * row_stride + col_offs, y, mask=mask)


def temperature_softmax(x: torch.Tensor, tau: float) -> torch.Tensor:
    assert tau > 0
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _temp_softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, tau, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


# Exercise 3: Log-softmax --------------------------------------------------

@triton.jit
def _log_softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols
    x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    shifted = x - m
    e = tl.where(mask, tl.exp(shifted), 0.0)
    denom = tl.sum(e, axis=0)
    y = shifted - tl.log(denom)
    tl.store(y_ptr + row_idx * row_stride + col_offs, y, mask=mask)


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _log_softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


# Exercise 4: Variable-length rows -----------------------------------------

@triton.jit
def _varlen_softmax_kernel(x_ptr, y_ptr, seq_lens_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + row_idx)
    row_start = x_ptr + row_idx * row_stride
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < seq_len
    x = tl.load(row_start + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    y = e / tl.sum(e, axis=0)
    # Padded positions written as 0.
    write_mask = col_offs < n_cols
    out = tl.where(mask, y, 0.0)
    tl.store(y_ptr + row_idx * row_stride + col_offs, out, mask=write_mask)


def varlen_softmax(x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.zeros_like(x)
    _varlen_softmax_kernel[(n_rows,)](x, y, seq_lens, x.stride(0), n_cols, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


# Exercise 5: Column softmax ----------------------------------------------

@triton.jit
def _col_softmax_kernel(x_ptr, y_ptr, M, N, row_stride, BLOCK_M: tl.constexpr):
    col_idx = tl.program_id(0)
    row_offs = tl.arange(0, BLOCK_M)
    mask = row_offs < M
    # Strided loads: one element per row.
    ptrs = x_ptr + row_offs * row_stride + col_idx
    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row_offs * row_stride + col_idx, y, mask=mask)


def column_softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_M = triton.next_power_of_2(M)
    y = torch.empty_like(x)
    _col_softmax_kernel[(N,)](x, y, M, N, x.stride(0), BLOCK_M=BLOCK_M, num_warps=4)
    return y


# Exercise 6: see online_softmax.py for the two-pass variant. A single-pass
# variant that fits in registers is left below; it only works when the row
# fits in a single tile, which makes it equivalent to stable_softmax. The
# point of the exercise is to realize the streaming "single-pass" idea is a
# misnomer unless you stage to memory.


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda")
    print("causal max diff:", (causal_softmax(x) - torch.softmax(x.masked_fill(torch.triu(torch.ones(64, 64, device="cuda", dtype=torch.bool), diagonal=1)[:8], float("-inf")), dim=-1)).abs().max().item())
    print("temp max diff:", (temperature_softmax(x, 0.7) - torch.softmax(x / 0.7, dim=-1)).abs().max().item())
    print("log_softmax max diff:", (log_softmax(x) - torch.log_softmax(x, dim=-1)).abs().max().item())
    print("col_softmax max diff:", (column_softmax(x) - torch.softmax(x, dim=0)).abs().max().item())
