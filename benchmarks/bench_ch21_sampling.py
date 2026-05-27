"""Benchmark sampling kernels at decode shapes (N=1)."""

import torch
import triton

from ch21_sampling import temperature_scale, argmax_sample, top_k_mask


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    N = 1  # autoregressive decode
    print(f"N={N} decode token; medians in microseconds")
    print(f"{'V':>8} {'temp_t':>10} {'temp_pt':>10} {'argmax_t':>10} {'argmax_pt':>10} {'topk50_t':>10}")
    for V in [32000, 50272, 128000]:
        x = torch.randn(N, V, device="cuda", dtype=torch.float16)
        ms_temp_t = triton.testing.do_bench(lambda: temperature_scale(x, 0.8))
        ms_temp_p = triton.testing.do_bench(lambda: x / 0.8)
        ms_arg_t = triton.testing.do_bench(lambda: argmax_sample(x))
        ms_arg_p = triton.testing.do_bench(lambda: x.argmax(dim=-1))
        ms_topk = triton.testing.do_bench(lambda: top_k_mask(x, k=50))
        # convert to us for the eye
        us = lambda v: v * 1000
        print(f"{V:>8} {us(ms_temp_t):>10.2f} {us(ms_temp_p):>10.2f} "
              f"{us(ms_arg_t):>10.2f} {us(ms_arg_p):>10.2f} {us(ms_topk):>10.2f}")


if __name__ == "__main__":
    main()
