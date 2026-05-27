"""Solutions for Chapter 22 exercises (low precision basics).

These are pure PyTorch warmups; no Triton yet.
"""

import torch
import torch.nn.functional as F


# Exercise 1: per-tensor symmetric int8 round-trip.
def exercise_1():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    scale = x.abs().amax() / 127.0
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    x_hat = scale * q.to(torch.float32)
    err = (x - x_hat).abs().max().item()
    print(f"[ex1] per-tensor symmetric int8 max abs error: {err:.6f}")
    return err


# Exercise 2: per-channel symmetric int8.
# Why is the error smaller? A single scalar scale must cover the largest
# absolute value across the entire matrix, which wastes resolution on rows
# whose values are much smaller. Per-row scaling picks a tight scale for
# each row, so quiet rows are quantized at finer granularity.
def exercise_2():
    torch.manual_seed(0)
    W = torch.randn(64, 1024, dtype=torch.float32)
    scale = W.abs().amax(dim=1) / 127.0           # (64,)
    q = torch.round(W / scale[:, None]).clamp(-128, 127).to(torch.int8)
    W_hat = scale[:, None] * q.to(torch.float32)
    err = (W - W_hat).abs().max().item()
    print(f"[ex2] per-channel symmetric int8 max abs error: {err:.6f}")
    return err


# Exercise 3: asymmetric int8 with zero-point on a non-negative tensor.
# Symmetric int8 maps [-127, 127] onto [-|max|, |max|] which is twice the
# range a non-negative tensor actually uses. The asymmetric layout shifts
# zero-point so the full 256-step grid covers [min, max], giving roughly
# half the quantization step size and roughly half the round-trip error.
def exercise_3():
    torch.manual_seed(0)
    x = F.relu(torch.randn(1024, dtype=torch.float32))

    # asymmetric
    scale_a = (x.max() - x.min()) / 255.0
    zp = torch.round(-x.min() / scale_a).to(torch.int32)
    q_a = torch.round(x / scale_a + zp).clamp(0, 255).to(torch.int32)
    x_hat_a = scale_a * (q_a.to(torch.float32) - zp.to(torch.float32))
    err_a = (x - x_hat_a).abs().max().item()

    # symmetric for comparison
    scale_s = x.abs().amax() / 127.0
    q_s = torch.round(x / scale_s).clamp(-128, 127).to(torch.int8)
    x_hat_s = scale_s * q_s.to(torch.float32)
    err_s = (x - x_hat_s).abs().max().item()

    print(f"[ex3] asymmetric max abs error: {err_a:.6f}")
    print(f"[ex3] symmetric  max abs error: {err_s:.6f}")
    return err_a, err_s


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
