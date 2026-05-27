# Chapter 22 exercises

Chapter 22 is a reference chapter; these three exercises are optional warmups in plain PyTorch to fix the quantize/dequant math in muscle memory before Chapter 23 puts it into Triton.

## Beginner

1. **Per-tensor symmetric int8 round-trip.** For a random fp32 vector `x` of length 1024, implement `q = round(x / scale).clamp(-128, 127).to(int8)` with `scale = x.abs().amax() / 127`. Dequantize and report `max((x - x_hat).abs())`.

   *Hint:* a single scalar scale; broadcasting is trivial.

2. **Per-channel symmetric int8.** Repeat for a `(64, 1024)` matrix `W` with axis=1 as the in-features axis and one scale per row. The scale tensor has shape `(64,)` and broadcasts along axis=1 in the divide. The round-trip error should be noticeably smaller than per-tensor. Explain in two sentences why.

   *Hint:* `W.abs().amax(dim=1)` returns shape `(64,)`. Use `scale[:, None]` to broadcast.

## Intermediate

3. **Asymmetric int8 with zero-point.** Implement `q = round(x / scale + zp).clamp(0, 255).to(uint8)` and the matching dequant `x_hat = scale * (q.float() - zp)` for a non-negative tensor `x = F.relu(torch.randn(1024))`. Choose `scale = (x.max() - x.min()) / 255` and `zp = round(-x.min() / scale)`. Compare the max round-trip error against symmetric int8 on the same tensor and explain which one wins for ReLU-shaped data.

   *Hint:* symmetric wastes half its range on negative values when the data is `>= 0`.
