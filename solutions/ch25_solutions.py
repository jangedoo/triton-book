"""Solutions for Chapter 25 — benchmarking."""

import time

import torch
import triton
import triton.language as tl

from src.ch25_benchmarking import bench, bench_stats, bandwidth_gbs, compare


# ---------------------------------------------------------------------------
# Exercise 1: vector_add bandwidth.
# ---------------------------------------------------------------------------
@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def vector_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    _vector_add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


def exercise_1():
    for log_n in (16, 18, 20, 22, 24):
        n = 1 << log_n
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        ms = bench(lambda: vector_add(x, y))
        gbs = bandwidth_gbs(3 * n * 4, ms)
        print(f"N=2^{log_n:<2}  {ms:8.4f} ms   {gbs:7.1f} GB/s")


# ---------------------------------------------------------------------------
# Exercise 2: the misleading benchmark fixed.
# Three things wrong:
#   1) No warmup — the first call JIT-compiles or pages in.
#   2) No cuda sync before stopping the clock — measures launch latency only.
#   3) Single shot — high variance, no averaging.
# ---------------------------------------------------------------------------
def bench_softmax_fixed(N):
    x = torch.randn(1, N, device="cuda")
    return bench(lambda: torch.softmax(x, dim=-1))


# ---------------------------------------------------------------------------
# Exercise 3: PyTorch eager vs compiled vs Triton softmax.
# (We use torch.softmax stand-ins for the Triton softmax to keep the file
#  self-contained; in your project, import from ch05.)
# ---------------------------------------------------------------------------
def exercise_3():
    torch_compile_softmax = torch.compile(lambda x: torch.softmax(x, dim=-1))
    for N in (1024, 4096, 16384):
        x = torch.randn(1024, N, device="cuda")
        # warm the compiled path
        for _ in range(3):
            torch_compile_softmax(x)
        results = compare({
            "eager":    lambda: torch.softmax(x, dim=-1),
            "compiled": lambda: torch_compile_softmax(x),
        })
        print(f"N={N}: {results}")


# ---------------------------------------------------------------------------
# Exercise 4: launch overhead measurement.
# ---------------------------------------------------------------------------
@triton.jit
def _noop_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    tl.store(x_ptr + pid, x)   # round-trip 1 element


def exercise_4():
    x = torch.zeros(1, device="cuda", dtype=torch.float32)
    ms = bench(lambda: _noop_kernel[(1,)](x, BLOCK=1))
    print(f"single-program launch overhead: {ms * 1000:.2f} microseconds")


# ---------------------------------------------------------------------------
# Exercise 5: bench_stats for softmax.
# ---------------------------------------------------------------------------
def exercise_5():
    x = torch.randn(1024, 4096, device="cuda")
    stats = bench_stats(lambda: torch.softmax(x, dim=-1))
    spread = stats["max"] - stats["min"]
    print(f"min={stats['min']:.3f} med={stats['median']:.3f} "
          f"mean={stats['mean']:.3f} max={stats['max']:.3f} "
          f"spread={spread:.3f} ms")
    # Spread comes from GPU clock throttling, contention with other apps,
    # and DRAM refresh cycles. Median is the right summary statistic.


# ---------------------------------------------------------------------------
# Exercise 6: torch.profiler sketch.
# Run interactively, not as a unit-testable function.
# ---------------------------------------------------------------------------
def exercise_6_sketch():
    from torch.profiler import profile, ProfilerActivity

    B, H, S, D = 2, 8, 512, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # warm
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p:
        for _ in range(50):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    exercise_1()
    print()
    exercise_3()
    print()
    exercise_4()
    print()
    exercise_5()
