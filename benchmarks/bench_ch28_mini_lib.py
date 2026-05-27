"""Unified benchmark for every kernel in `mini_triton_llm`.

Runs each Triton kernel and its PyTorch reference at one representative
shape, then prints a table via `mini_triton_llm.benchmarking.compare`.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "ch28_mini_lib"),
)

from mini_triton_llm import (  # noqa: E402
    cross_entropy,
    flash_attention,
    residual_rmsnorm,
    rmsnorm,
    rope,
    softmax,
    swiglu,
)
from mini_triton_llm.benchmarking import compare, print_table  # noqa: E402
from mini_triton_llm.testing import random_qkv  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        print("cuda only; skipping")
        return

    torch.manual_seed(0)
    rows = []

    # RMSNorm: (M, N) = (4096, 4096) fp16
    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(4096, dtype=torch.float16, device="cuda")

    def torch_rms() -> torch.Tensor:
        r = torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
        return (x.float() * r * w.float()).to(torch.float16)

    rows.append(compare("rmsnorm", lambda: rmsnorm(x, w), torch_rms))

    # Residual + RMSNorm
    r = torch.randn_like(x)

    def torch_res_rms() -> torch.Tensor:
        h = (x + r).float()
        s = torch.rsqrt((h ** 2).mean(-1, keepdim=True) + 1e-6)
        return (h * s * w.float()).to(torch.float16)

    rows.append(compare("residual_rmsnorm", lambda: residual_rmsnorm(x, r, w)[1], torch_res_rms))

    # Softmax
    rows.append(compare("softmax", lambda: softmax(x), lambda: torch.softmax(x.float(), dim=-1).half()))

    # RoPE
    B, S, H, D = 4, 1024, 8, 64
    xr = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    pos = torch.arange(S, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    freqs = pos[:, None] * inv_freq[None, :]
    cos = freqs.cos().half()
    sin = freqs.sin().half()

    def torch_rope() -> torch.Tensor:
        x0, x1 = xr[..., : D // 2], xr[..., D // 2 :]
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
        return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)

    rows.append(compare("rope", lambda: rope(xr, cos, sin), torch_rope))

    # SwiGLU
    a = torch.randn(4096, 11008, dtype=torch.float16, device="cuda")
    b = torch.randn_like(a)
    rows.append(compare("swiglu", lambda: swiglu(a, b), lambda: F.silu(a) * b))

    # Flash attention (small to fit a 2070 SUPER)
    q, k, v = random_qkv(2, 4, 1024, 64, dtype=torch.float16)
    rows.append(
        compare(
            "flash_attention",
            lambda: flash_attention(q, k, v, causal=True),
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        )
    )

    # Cross entropy
    N, V = 4096, 32000
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    tgts = torch.randint(0, V, (N,), device="cuda", dtype=torch.long)
    rows.append(
        compare(
            "cross_entropy",
            lambda: cross_entropy(logits, tgts),
            lambda: F.cross_entropy(logits, tgts),
        )
    )

    print_table(rows)


if __name__ == "__main__":
    main()
