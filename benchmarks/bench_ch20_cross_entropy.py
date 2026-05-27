"""Benchmark fused cross-entropy across modern LLM vocab sizes."""

import torch
import torch.nn.functional as F
import triton

from ch20_cross_entropy import cross_entropy_forward


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    N = 4096
    print(f"N={N} rows; medians in ms")
    print(f"{'V':>8} {'triton':>10} {'torch':>10}")
    for V in [32000, 50272, 128000, 256000]:
        logits = torch.randn(N, V, device="cuda", dtype=torch.float16)
        target = torch.randint(0, V, (N,), device="cuda")
        t = triton.testing.do_bench(lambda: cross_entropy_forward(logits, target))
        p = triton.testing.do_bench(lambda: F.cross_entropy(logits.float(), target))
        print(f"{V:>8} {t:>10.4f} {p:>10.4f}")


if __name__ == "__main__":
    main()
