"""Benchmark fused SwiGLU/GEGLU epilogue against eager-PyTorch."""

import torch
import torch.nn.functional as F
import triton

from ch19_fused_swiglu import swiglu_bias, geglu_bias


def torch_swiglu(xg, bg, xu, bu):
    return F.silu(xg + bg) * (xu + bu)


def torch_geglu(xg, bg, xu, bu):
    return F.gelu(xg + bg, approximate="tanh") * (xu + bu)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    M = 4096
    print(f"M={M} rows; medians in ms")
    print(f"{'H':>6} {'swiglu_triton':>16} {'swiglu_torch':>16} {'geglu_triton':>16} {'geglu_torch':>16}")
    for H in [4096, 8192, 11008, 14336]:
        xg = torch.randn(M, H, device="cuda", dtype=torch.float16)
        bg = torch.randn(H,    device="cuda", dtype=torch.float16)
        xu = torch.randn(M, H, device="cuda", dtype=torch.float16)
        bu = torch.randn(H,    device="cuda", dtype=torch.float16)
        st = triton.testing.do_bench(lambda: swiglu_bias(xg, bg, xu, bu))
        sp = triton.testing.do_bench(lambda: torch_swiglu(xg, bg, xu, bu))
        gt = triton.testing.do_bench(lambda: geglu_bias(xg, bg, xu, bu))
        gp = triton.testing.do_bench(lambda: torch_geglu(xg, bg, xu, bu))
        print(f"{H:>6} {st:>16.4f} {sp:>16.4f} {gt:>16.4f} {gp:>16.4f}")


if __name__ == "__main__":
    main()
