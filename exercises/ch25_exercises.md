# Chapter 25 exercises — benchmarking

## Beginner

1. **vector_add bandwidth.** Take the `vector_add` kernel from Chapter 2. Benchmark it for `N in [2**16, 2**18, 2**20, 2**22, 2**24]`. Use `bench()` from `src/ch25_benchmarking/benchmark_utils.py`. For each size, compute and print GB/s. Plot it.

   *Hint:* bytes = 3 * N * sizeof(fp32) (two reads + one write).

2. **The misleading benchmark.** The following snippet looks reasonable but lies. Identify the three things wrong with it and fix them.
   ```python
   def bench_softmax(N):
       x = torch.randn(1, N, device="cuda")
       t0 = time.perf_counter()
       y = torch.softmax(x, dim=-1)
       t1 = time.perf_counter()
       return (t1 - t0) * 1000   # ms
   ```
   *Hint:* warmup, sync, single shot.

3. **PyTorch eager vs compiled vs Triton softmax.** Use the Triton softmax from Chapter 5. Compare against `torch.softmax` (eager) and `torch.compile(torch.softmax)` for `N in [1024, 4096, 16384]`. Use `compare(...)`. Print a table.

   *Hint:* warm up the compiled version separately — `torch.compile` JITs on first call.

## Intermediate

4. **Launch overhead.** Write a Triton kernel that does nothing (one `tl.load` of one element, one `tl.store` of one element). Benchmark a single launch. The number you get is the launch overhead (typically 5–15 microseconds on a 2070 SUPER). Use this to argue against benchmarking single small launches.

   *Hint:* if your kernel runs in 2 microseconds, you cannot measure it accurately because the launch alone is 10x longer.

5. **bench_stats helper.** Use `bench_stats()` to report min, median, mean, and max latency for the Chapter 5 softmax at `M=1024, N=4096`. Discuss the spread (max - min) and what causes it.

   *Hint:* GPU contention, clock variation, page faults on first touches.

## Advanced

6. **torch.profiler.** Wrap a Chapter 14 FlashAttention forward pass in `torch.profiler.profile(...)` with `record_shapes=True`. Identify the dominant kernel and the dominant memory op. Compare against a naive attention pass and report which ops disappear.

   *Hint:* `with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as p: ...; print(p.key_averages().table(sort_by="cuda_time_total"))`.
