"""Worked solutions for Chapter 4 exercises."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise B1: row_prod
# ---------------------------------------------------------------------------
@triton.jit
def row_prod_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * sxm
    acc = tl.full((), 1.0, dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        # other=1.0: multiplicative identity, so masked lanes do not change the product.
        x = tl.load(x_row + cols * sxn, mask=mask, other=1.0).to(tl.float32)
        # Reduce within tile, then combine.
        tile_prod = tl.reduce(x, axis=0, combine_fn=lambda a, b: a * b)
        acc = acc * tile_prod
    tl.store(out_ptr + pid_m, acc)


def row_prod(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    row_prod_kernel[(M,)](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


# ---------------------------------------------------------------------------
# Exercise B2: row_l2
# ---------------------------------------------------------------------------
@triton.jit
def row_l2_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * sxm
    acc = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * sxn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(out_ptr + pid_m, tl.sqrt(acc))


def row_l2(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    row_l2_kernel[(M,)](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


# ---------------------------------------------------------------------------
# Exercise B3: row_argmax (single pass — assumes N <= BLOCK_SIZE_N for simplicity)
# ---------------------------------------------------------------------------
@triton.jit
def row_argmax_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * sxm
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x = tl.load(x_row + cols * sxn, mask=mask, other=float("-inf")).to(tl.float32)
    # tl.argmax exists in recent Triton; fall back to argmax-via-reduce if not.
    idx = tl.argmax(x, axis=0)
    tl.store(out_ptr + pid_m, idx)


def row_argmax(x: torch.Tensor, BLOCK_SIZE_N: int | None = None) -> torch.Tensor:
    """Single-tile argmax. BLOCK_SIZE_N defaults to the next power of two >= N."""
    M, N = x.shape
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
    out = torch.empty(M, device=x.device, dtype=torch.int32)
    row_argmax_kernel[(M,)](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


# ---------------------------------------------------------------------------
# Exercise I1: column-wise sum
# ---------------------------------------------------------------------------
@triton.jit
def col_sum_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for row in range(0, M):
        x = tl.load(x_ptr + row * sxm + cols * sxn, mask=col_mask, other=0.0).to(tl.float32)
        acc += x
    tl.store(out_ptr + cols, acc, mask=col_mask)


def col_sum(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    col_sum_kernel[grid](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


# ---------------------------------------------------------------------------
# Exercise I2: row_logsumexp (two-pass)
# ---------------------------------------------------------------------------
@triton.jit
def row_logsumexp_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * sxm

    # Pass 1: max.
    NEG_INF = float("-inf")
    m = tl.full((), NEG_INF, dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * sxn, mask=mask, other=NEG_INF).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    # Pass 2: sum of shifted exps.
    s = tl.zeros((), dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * sxn, mask=mask, other=NEG_INF).to(tl.float32)
        s += tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=0)

    tl.store(out_ptr + pid_m, tl.log(s) + m)


def row_logsumexp(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    row_logsumexp_kernel[(M,)](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N)
    return out


# ---------------------------------------------------------------------------
# Exercise A1: one-pass Welford variance
# ---------------------------------------------------------------------------
@triton.jit
def row_variance_welford_kernel(x_ptr, out_ptr, M, N, sxm, sxn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    x_row = x_ptr + pid_m * sxm

    # Running (count, mean, M2) for the row.
    n_acc = tl.zeros((), dtype=tl.float32)
    mean_acc = tl.zeros((), dtype=tl.float32)
    m2_acc = tl.zeros((), dtype=tl.float32)

    for col_start in range(0, N, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_row + cols * sxn, mask=mask, other=0.0).to(tl.float32)

        # Tile-local Welford on the lanes that are in bounds.
        in_bounds = mask.to(tl.float32)
        n_b = tl.sum(in_bounds, axis=0)
        # Use masked sums for mean / M2 so out-of-bounds lanes contribute 0.
        sum_b = tl.sum(tl.where(mask, x, 0.0), axis=0)
        mean_b = tl.where(n_b > 0, sum_b / n_b, 0.0)
        diff_b = tl.where(mask, x - mean_b, 0.0)
        m2_b = tl.sum(diff_b * diff_b, axis=0)

        # Combine partial (n_b, mean_b, m2_b) with the running (n_acc, mean_acc, m2_acc).
        n_new = n_acc + n_b
        delta = mean_b - mean_acc
        mean_new = tl.where(n_new > 0, mean_acc + delta * n_b / n_new, 0.0)
        m2_new = m2_acc + m2_b + delta * delta * n_acc * n_b / tl.where(n_new > 0, n_new, 1.0)

        n_acc = n_new
        mean_acc = mean_new
        m2_acc = m2_new

    tl.store(out_ptr + pid_m, m2_acc / n_acc)


def row_variance_welford(x: torch.Tensor, BLOCK_SIZE_N: int = 1024) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    row_variance_welford_kernel[(M,)](
        x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return out
