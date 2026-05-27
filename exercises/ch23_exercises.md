# Chapter 23 exercises

## Beginner

1. **Per-tensor dequant.** Strip `dequant_int8_per_channel` down to a per-tensor scale (a single scalar). The kernel becomes simpler — no scale load per row, no broadcast. Test on a `(256, 256)` int8 input.

   *Hint:* pass the scalar as a kernel argument, not a pointer.

2. **Per-channel dequant with axis=1.** Implement a version where the scale vector has shape `(N,)` and broadcasts along the M dimension instead of N. Useful when your weight matrix is stored with output channels as the second axis.

   *Hint:* `s = tl.load(scale_ptr + offs_n, ...)` and `s[None, :] * q`.

3. **Fused dequant + bias only.** Drop the GELU from `dequant_bias_gelu_fused`. This is the common LLM-inference pattern of "dequantize the weight + add a bias" before a separate activation.

   *Hint:* delete the GELU lines and keep everything else.

## Intermediate

4. **W8A16 matmul forward.** Read `w8a16_matmul.py` until you can re-derive it from scratch. Then change the scale layout from per-column (shape `(N,)`) to per-row of the **transposed** weight (shape `(N,)` but interpret the weight as stored `(N, K)` row-major). The dequant becomes a per-output-channel scale on a transposed load.

   *Hint:* swap the strides on `w` and re-derive the broadcast axis for `s`.

5. **W4A16 matmul sketch.** In int4 layout, two int4 values pack into a single int8 byte. Sketch a Triton kernel that loads `BLOCK_K * BLOCK_N // 2` bytes per weight tile and unpacks each byte into two int4 values with bit shifts: `low = (q & 0xF)` and `high = (q >> 4) & 0xF`. Then sign-extend (if signed int4) and proceed exactly like W8A16. You do not need to run it — produce the kernel and walk through the pointer math.

   *Hint:* `BLOCK_K` is the unpacked tile size; the byte loop iterates over `BLOCK_K // 2` bytes.

## Advanced

6. **W8A8 with int8 tensor cores.** sm_75 supports `tl.dot` with int8 inputs and an int32 accumulator. Write `w8a8_matmul` that takes int8 `x` and int8 `w`, accumulates into int32, multiplies by `x_scale * w_scale` at the end, and stores fp16. Test correctness against an fp32 reference. Benchmark against `w8a16_matmul` and report which one wins at `M=N=K=4096`.

   *Hint:* `tl.dot(x_int8, w_int8, out_dtype=tl.int32)`. The output scale is a single `(N,)` vector times a per-row activation scale `(M,)`, both applied after the dot.
