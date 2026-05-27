"""Benchmarks for Chapter 23.

Measures:
    1) dequant_int8_per_channel GB/s vs a naive PyTorch dequant.
    2) w8a16_matmul vs a fp16 reference matmul at memory-bound shapes
       (narrow M = 1, wide N).
"""

import torch
import triton

from src.ch23_quant import dequant_int8_per_channel, w8a16_matmul


def bench_dequant():
    print("=== dequant_int8_per_channel ===")
    print(f"{'M':>6} {'N':>6} {'triton_ms':>10} {'torch_ms':>10} {'triton_GB/s':>12}")
    for M, N in [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]:
        q = torch.randint(-128, 128, (M, N), dtype=torch.int8, device="cuda")
        scale = torch.rand(M, device="cuda", dtype=torch.float16) * 0.02 + 0.005

        def triton_fn():
            return dequant_int8_per_channel(q, scale)

        def torch_fn():
            return (scale[:, None].to(torch.float32) * q.to(torch.float32)).to(torch.float16)

        t_ms = triton.testing.do_bench(triton_fn, warmup=25, rep=100, return_mode="median")
        p_ms = triton.testing.do_bench(torch_fn, warmup=25, rep=100, return_mode="median")
        # bytes: read M*N int8 + M fp16 scale; write M*N fp16
        bytes_total = M * N * 1 + M * 2 + M * N * 2
        gbs = bytes_total / (t_ms * 1e-3) / 1e9
        print(f"{M:>6} {N:>6} {t_ms:>10.3f} {p_ms:>10.3f} {gbs:>12.1f}")


def bench_w8a16():
    print("\n=== w8a16_matmul vs fp16 matmul ===")
    print(f"{'M':>4} {'N':>6} {'K':>6} {'w8a16_ms':>9} {'fp16_ms':>9} {'speedup':>8}")
    # memory-bound shapes: small M (decode-like), large N, K.
    for M, N, K in [(1, 4096, 4096), (1, 8192, 8192),
                    (4, 4096, 4096), (16, 4096, 4096),
                    (1024, 1024, 1024)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        w_int8 = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
        scale = torch.rand(N, device="cuda", dtype=torch.float16) * 0.02 + 0.005
        w_fp16 = (scale[None, :].to(torch.float32) * w_int8.to(torch.float32)).to(torch.float16)

        def triton_fn():
            return w8a16_matmul(x, w_int8, scale)

        def torch_fn():
            return x @ w_fp16

        # warm both, then measure
        t_ms = triton.testing.do_bench(triton_fn, warmup=25, rep=100, return_mode="median")
        p_ms = triton.testing.do_bench(torch_fn, warmup=25, rep=100, return_mode="median")
        print(f"{M:>4} {N:>6} {K:>6} {t_ms:>9.3f} {p_ms:>9.3f} {p_ms / t_ms:>7.2f}x")


if __name__ == "__main__":
    bench_dequant()
    bench_w8a16()
