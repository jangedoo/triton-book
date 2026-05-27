"""Benchmark for Chapter 17: append_kv_cache throughput and
decode_attention scaling with cache length."""

import torch
import triton
import torch.nn.functional as F

from src.ch17_kv_cache.append_kv_cache import append_kv_cache
from src.ch17_kv_cache.decode_attention import decode_attention


def bench_append(B, H, D, max_seq, dtype):
    k_new = torch.randn(B, H, 1, D, device="cuda", dtype=dtype)
    v_new = torch.randn_like(k_new)
    k_cache = torch.empty(B, H, max_seq, D, device="cuda", dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    ms = triton.testing.do_bench(lambda: append_kv_cache(k_new, v_new, k_cache, v_cache, 0))
    bytes_moved = 2 * B * H * D * k_new.element_size()  # read + write of K + V
    return ms, bytes_moved


def bench_decode(B, H, D, max_seq, current_len, dtype):
    q = torch.randn(B, H, 1, D, device="cuda", dtype=dtype)
    k_cache = torch.randn(B, H, max_seq, D, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)

    triton_ms = triton.testing.do_bench(lambda: decode_attention(q, k_cache, v_cache, current_len))

    def torch_ref():
        return F.scaled_dot_product_attention(q, k_cache[:, :, :current_len, :], v_cache[:, :, :current_len, :])

    torch_ms = triton.testing.do_bench(torch_ref)
    cache_bytes = 2 * B * H * current_len * D * k_cache.element_size()
    return triton_ms, torch_ms, cache_bytes


def main():
    assert torch.cuda.is_available(), "cuda only"

    print("=== append_kv_cache ===")
    print(f"{'shape':>30}  {'ms':>10}  {'GB/s':>10}")
    for B in (1, 4, 16, 64):
        H, D, max_seq = 32, 128, 8192
        ms, bytes_moved = bench_append(B, H, D, max_seq, torch.float16)
        tag = f"B={B} H={H} D={D}"
        print(f"{tag:>30}  {ms:>10.4f}  {bytes_moved/ms/1e6:>10.1f}")

    print("\n=== decode_attention ===")
    print(f"{'cache_len':>12}  {'triton ms':>12}  {'torch ms':>12}  {'triton GB/s':>14}")
    B, H, D, max_seq = 1, 32, 128, 16384
    for cl in (128, 512, 2048, 8192, 16384):
        triton_ms, torch_ms, cache_bytes = bench_decode(B, H, D, max_seq, cl, torch.float16)
        print(f"{cl:>12}  {triton_ms:>12.3f}  {torch_ms:>12.3f}  {cache_bytes/triton_ms/1e6:>14.1f}")


if __name__ == "__main__":
    main()
