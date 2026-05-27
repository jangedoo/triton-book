"""Solutions for Chapter 26 — autotuning."""

import json
import os

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Exercise 1: autotuned softmax over BLOCK_SIZE.
# (See src/ch26_autotune/autotuned_softmax.py for the canonical version.)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in (128, 256, 512, 1024, 2048)],
    key=["N"],
)
@triton.jit
def softmax_kernel_ex1(
    x_ptr, y_ptr, stride_xm, stride_ym, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(tl.float32))
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + pid * stride_ym + offs, (e / s).to(x.dtype), mask=mask)


def softmax_ex1(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel_ex1[(M,)](x, y, x.stride(0), y.stride(0), M, N)
    return y


# ---------------------------------------------------------------------------
# Exercise 2: autotuned matmul over a 4-config grid.
# (See src/ch26_autotune/autotuned_matmul.py for a fuller config list.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exercise 3: triton.heuristics for BLOCK_SIZE.
# Heuristics computes a constexpr from kernel args at launch time. No tuning
# happens. Prefer heuristics when:
#   - the right BLOCK_SIZE is a deterministic function of the input;
#   - you don't want to pay the first-call autotune cost;
#   - num_warps doesn't matter for your kernel.
# Prefer autotune when:
#   - the right config depends on the GPU as well as the shape;
#   - num_warps or num_stages matter;
#   - the search space is non-trivial.
# ---------------------------------------------------------------------------
@triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def softmax_kernel_heuristic(
    x_ptr, y_ptr, stride_xm, stride_ym, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(tl.float32))
    s = tl.sum(e, axis=0)
    tl.store(y_ptr + pid * stride_ym + offs, (e / s).to(x.dtype), mask=mask)


def softmax_heuristic(x):
    M, N = x.shape
    y = torch.empty_like(x)
    softmax_kernel_heuristic[(M,)](x, y, x.stride(0), y.stride(0), M, N)
    return y


# ---------------------------------------------------------------------------
# Exercise 4: show the picker changes with shape.
# ---------------------------------------------------------------------------
def exercise_4():
    from src.ch26_autotune.autotuned_matmul import _matmul_kernel, autotuned_matmul
    for M, N, K in [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        _ = autotuned_matmul(a, b)
        # best_config is a dict keyed on the autotune `key` tuple
        print(f"({M},{N},{K}) -> {_matmul_kernel.best_config}")


# ---------------------------------------------------------------------------
# Exercise 5: persist autotune results.
# triton.autotune stores the chosen Config per key in `_matmul_kernel.cache`.
# We can serialize the dict ourselves on shutdown and re-seed it on next run.
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.expanduser("~/.cache/triton_book/matmul_autotune.json")


def save_autotune_cache(kernel):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    serial = {}
    for key, cfg in kernel.cache.items():
        # cfg is a triton.Config; pull out the bits we need to rebuild it.
        serial[str(key)] = {
            "kwargs":     cfg.kwargs,
            "num_warps":  cfg.num_warps,
            "num_stages": cfg.num_stages,
        }
    with open(CACHE_PATH, "w") as f:
        json.dump(serial, f, indent=2)


def load_autotune_cache(kernel, configs):
    """Re-seed kernel.cache from a JSON file. Best-effort; if the file is
    missing or stale, the autotuner will simply re-search on first launch.
    """
    if not os.path.exists(CACHE_PATH):
        return
    with open(CACHE_PATH) as f:
        serial = json.load(f)
    # Match each serialized entry to one of the supplied Configs.
    cfg_index = {(tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages): c
                 for c in configs}
    for key_str, blob in serial.items():
        sig = (tuple(sorted(blob["kwargs"].items())),
               blob["num_warps"], blob["num_stages"])
        cfg = cfg_index.get(sig)
        if cfg is None:
            continue
        # The autotune cache key is a tuple; we stored it as str.
        kernel.cache[eval(key_str)] = cfg


# ---------------------------------------------------------------------------
# Exercise 6: autotune a FlashAttention forward (sketch).
# Wrap the FlashAttention kernel with:
#   @triton.autotune(
#       configs=[
#           triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4),
#           triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4),
#           triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8),
#           triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
#       ],
#       key=["H", "S", "D"],
#   )
# Then compare the chosen config to the hand-tuned one. On a 2070 SUPER you
# will typically see BLOCK_M=128, BLOCK_N=64, num_warps=4 win for D=64.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    y = softmax_ex1(x)
    print("ex1 ok:", torch.allclose(y, torch.softmax(x, dim=-1), atol=1e-4))
    y2 = softmax_heuristic(x)
    print("ex3 ok:", torch.allclose(y2, torch.softmax(x, dim=-1), atol=1e-4))
    exercise_4()
