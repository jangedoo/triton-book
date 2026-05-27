# Chapter 26 exercises — autotuning

## Beginner

1. **Autotune softmax over BLOCK_SIZE.** Take the Chapter 5 softmax and wrap it in `@triton.autotune` with configs `[128, 256, 512, 1024, 2048]`. Use `key=["N"]`. Benchmark at `N in [256, 1024, 4096]` and report the chosen config for each.

   *Hint:* one `triton.Config({"BLOCK_SIZE": v})` per value.

2. **Autotune matmul over a small grid.** Wrap the Chapter 9 matmul in autotune with the 4 configs `(BLOCK_M, BLOCK_N, BLOCK_K) in [(64, 64, 32), (128, 64, 32), (64, 128, 32), (128, 128, 32)]`. Key on `(M, N, K)`. Benchmark at `(512, 512, 512)` and `(2048, 2048, 2048)` and report the chosen config for each.

   *Hint:* add `num_warps=2` for small blocks, `num_warps=4` for larger.

3. **triton.heuristics for BLOCK_SIZE.** Add `@triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["N"])})` above the softmax `@triton.jit` and **remove** the autotune decorator. Verify the kernel still works for `N in [100, 1000, 4000]` and explain when you would prefer heuristics over autotune.

   *Hint:* heuristics has zero search cost but cannot consider `num_warps`.

## Intermediate

4. **Show the autotune picker changes with shape.** Write a script that calls the autotuned matmul for three shapes (small, medium, large), then prints `_matmul_kernel.best_config` and observes that it differs per shape.

   *Hint:* the `best_config` attribute on the decorated kernel maps (shape key) → chosen config.

5. **Persist autotune results.** Write a wrapper that serializes the autotune cache to disk (a JSON or pickle file) and reloads it on the next run, so you only pay search cost once per machine. The cache is `_matmul_kernel.cache`.

   *Hint:* the cache key is a tuple matching the `key=` arg; the value is a `triton.Config`.

## Advanced

6. **Autotune a fused attention kernel.** Take a FlashAttention forward from Chapter 14 and autotune over `(BLOCK_M, BLOCK_N, num_warps)` for `(BLOCK_M, BLOCK_N) in [(64, 64), (128, 64), (64, 128), (128, 128)]` and `num_warps in [4, 8]`. Use `key=["H", "S", "D"]` (head count, sequence length, head dim). Compare against the Chapter 14 hand-tuned config.

   *Hint:* attention configs are sensitive to head dim — restrict `BLOCK_N` to multiples of 16 and watch the autotuner avoid the spill-heavy ones.
