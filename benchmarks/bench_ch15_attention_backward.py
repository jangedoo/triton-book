"""Minimal benchmark for the educational attention backward.

This kernel is not performance-tuned. We log a single timing point so the
reader can see the order of magnitude. Real Flash backward is ~3x slower than
its forward; the educational backward here is much slower than that.
"""

import torch
import triton
import torch.nn.functional as F

from src.ch15_attention_backward.educational_backward import attention_backward_educational


def main():
    assert torch.cuda.is_available(), "cuda only"
    B, H, S, D = 1, 8, 256, 64
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    do = torch.randn_like(q)

    edu_ms = triton.testing.do_bench(
        lambda: attention_backward_educational(q, k, v, do)
    )

    def torch_bwd():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        o = F.scaled_dot_product_attention(q_, k_, v_)
        torch.autograd.grad(o, [q_, k_, v_], do)

    torch_ms = triton.testing.do_bench(torch_bwd)

    print(f"shape: B={B} H={H} S={S} D={D}")
    print(f"  educational backward: {edu_ms:.3f} ms")
    print(f"  torch SDPA backward : {torch_ms:.3f} ms")
    print(f"  ratio (edu / torch) : {edu_ms / torch_ms:.1f}x")


if __name__ == "__main__":
    main()
