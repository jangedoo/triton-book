"""Benchmark for Chapter 16: standalone RoPE kernel vs PyTorch reference.

RoPE is memory-bound. The expected story is that Triton roughly matches the
DRAM ceiling; the bigger wins come later from fusing RoPE into the Q/K
projection (preview of Chapters 18-19).
"""

import torch
import triton

from src.ch16_rope.rope import build_cos_sin_cache, rope_noninterleaved


def torch_rope_noninterleaved(x, cos, sin):
    D = x.shape[-1]
    x1 = x[..., : D // 2].float()
    x2 = x[...,   D // 2 :].float()
    c = cos[None, None, :, :]; s = sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


def bench_one(B, H, S, D, dtype):
    x = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    cos, sin = build_cos_sin_cache(max_seq=S, dim=D, device="cuda")

    triton_ms = triton.testing.do_bench(lambda: rope_noninterleaved(x, cos, sin))
    torch_ms  = triton.testing.do_bench(lambda: torch_rope_noninterleaved(x, cos, sin))

    bytes_moved = 2 * x.numel() * x.element_size()  # one read + one write
    triton_bw = bytes_moved / triton_ms / 1e6  # GB/s
    return triton_ms, torch_ms, triton_bw


def main():
    assert torch.cuda.is_available(), "cuda only"
    print(f"{'shape':>30}  {'triton (ms)':>12}  {'torch (ms)':>12}  {'triton GB/s':>14}")
    for S in (512, 1024, 2048, 4096):
        B, H, D = 2, 32, 128
        triton_ms, torch_ms, bw = bench_one(B, H, S, D, torch.float16)
        tag = f"B={B} H={H} S={S} D={D}"
        print(f"{tag:>30}  {triton_ms:>12.3f}  {torch_ms:>12.3f}  {bw:>14.1f}")


if __name__ == "__main__":
    main()
