"""Autotuned row-wise softmax.

Picks BLOCK_SIZE per N. Also demonstrates triton.heuristics for setting
BLOCK_SIZE from the input shape when you don't want to tune it.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["N"],   # pick a config per distinct N
)
@triton.jit
def _softmax_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_ym,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(tl.float32))
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(x.dtype)

    tl.store(y_ptr + pid * stride_ym + offs, y, mask=mask)


def autotuned_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax with autotuned BLOCK_SIZE per N.

    Requires N <= 4096 with the current config list. Add larger configs to
    support wider rows.
    """
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    assert N <= 4096, "extend the autotune configs for wider rows"
    y = torch.empty_like(x)
    _softmax_kernel[(M,)](
        x, y,
        x.stride(0), y.stride(0),
        M, N,
    )
    return y


# ---------------------------------------------------------------------------
# Variant: triton.heuristics instead of full autotuning.
# Used when you don't want to pay autotune cost; you just want BLOCK_SIZE to
# track N at launch time.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def _softmax_kernel_heuristic(
    x_ptr, y_ptr,
    stride_xm, stride_ym,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(tl.float32))
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(x.dtype)
    tl.store(y_ptr + pid * stride_ym + offs, y, mask=mask)


def softmax_heuristic(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _softmax_kernel_heuristic[(M,)](
        x, y, x.stride(0), y.stride(0), M, N,
    )
    return y
