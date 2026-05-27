"""Benchmark fused residual + RMSNorm vs two-op PyTorch and torch.compile."""

import torch
import triton

from ch18_residual_rmsnorm import residual_rmsnorm


def torch_two_ops(x, r, w, eps=1e-6):
    t = x + r
    t32 = t.to(torch.float32)
    var = t32.pow(2).mean(dim=-1, keepdim=True)
    y = (t32 * torch.rsqrt(var + eps)).to(x.dtype) * w
    return y, t


torch_compiled = torch.compile(torch_two_ops)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    M = 4096
    print(f"M={M} rows; medians in ms")
    print(f"{'N':>6} {'triton':>10} {'torch_eager':>14} {'torch_compile':>14}")
    for N in [1024, 2048, 4096, 8192]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        r = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        triton_ms = triton.testing.do_bench(lambda: residual_rmsnorm(x, r, w))
        torch_ms = triton.testing.do_bench(lambda: torch_two_ops(x, r, w))
        compiled_ms = triton.testing.do_bench(lambda: torch_compiled(x, r, w))
        print(f"{N:>6} {triton_ms:>10.4f} {torch_ms:>14.4f} {compiled_ms:>14.4f}")


if __name__ == "__main__":
    main()
