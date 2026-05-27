# Chapter 27 exercises — diagnosis puzzles

Three numbered scenarios. For each, the question is the same: is the kernel **memory-bound** or **compute-bound**, and what is the next experiment to run? Answers in `solutions/ch27_solutions.py`.

Take peak DRAM bandwidth as ~448 GB/s and peak fp16 tensor-core throughput as ~57 TFLOP/s (2070 SUPER numbers; substitute your own GPU's specs if different).

## Puzzle 1

A row-wise layernorm kernel on a `(1024, 4096)` fp16 input takes 0.40 ms. Bytes moved: `1024 * 4096 * 2` (read) + `1024 * 4096 * 2` (write) + `4096 * 2 * 2` (weight + bias).

- Compute the GB/s achieved.
- Memory- or compute-bound?
- What's the next experiment?

## Puzzle 2

A square matmul of `(2048, 2048, 2048)` fp16 takes 3.5 ms. Total FLOPs: `2 * 2048**3`.

- Compute TFLOP/s.
- Memory- or compute-bound?
- What's the next experiment?

## Puzzle 3

A W8A16 decode matmul `(M=1, K=4096, N=4096)` takes 0.18 ms. Bytes moved (weight-dominated): `4096 * 4096 * 1` (int8 weight) + `4096 * 2` (fp16 activation) + `4096 * 2` (fp16 output).

- Compute GB/s.
- Memory- or compute-bound?
- What's the next experiment?
