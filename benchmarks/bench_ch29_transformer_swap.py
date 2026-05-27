"""Latency + memory comparison for PyTorch vs Triton transformer block.

Runs forward at three sequence lengths and prints two tables:

    1. median ms per forward (lower is better).
    2. peak CUDA memory across the forward (lower is better).

The memory number is the headline: the Triton attention path does not
materialize the [S, S] score matrix, so its peak memory grows linearly
with S rather than quadratically.
"""

import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "ch28_mini_lib"))

from ch29_transformer_swap import BlockConfig, PyTorchBlock, TritonBlock  # noqa: E402
from mini_triton_llm.benchmarking import bench  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        print("cuda only; skipping")
        return

    cfg = BlockConfig()
    ref = PyTorchBlock(cfg, dtype=torch.float16).cuda().eval()
    tri = TritonBlock.from_pytorch(ref).cuda().eval()

    B = 4
    seqs = [512, 1024, 2048]

    print(f"{'seq':<8} {'torch ms':>12} {'triton ms':>12} {'speedup':>10}")
    print("-" * 46)
    lat_rows = []
    for S in seqs:
        x = torch.randn(B, S, cfg.hidden_dim, dtype=torch.float16, device="cuda") * 0.1
        with torch.no_grad():
            t_torch = bench(lambda: ref(x))
            t_tri = bench(lambda: tri(x))
        lat_rows.append((S, t_torch, t_tri))
        print(f"{S:<8} {t_torch:>12.3f} {t_tri:>12.3f} {t_torch / t_tri:>9.2f}x")

    print()
    print(f"{'seq':<8} {'torch MiB':>12} {'triton MiB':>12}")
    print("-" * 36)
    for S in seqs:
        x = torch.randn(B, S, cfg.hidden_dim, dtype=torch.float16, device="cuda") * 0.1
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = ref(x)
        torch.cuda.synchronize()
        mb_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = tri(x)
        torch.cuda.synchronize()
        mb_tri = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{S:<8} {mb_torch:>12.1f} {mb_tri:>12.1f}")


if __name__ == "__main__":
    main()
