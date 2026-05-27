# Chapter 24 exercises — debugging puzzles

Each puzzle ships a broken kernel. Your job is to find the bug class, name it, and fix it. Solutions in `solutions/ch24_solutions.py`.

## Beginner

1. **The OOB-store puzzle.** `src/ch24_debugging/debug_demo.py` contains `add_with_mask_bug`. Run it with `n=1000` and a `BLOCK=128`. Sometimes the test passes; sometimes a downstream tensor mysteriously changes. Identify the bug, fix it.

   *Hint:* compare `tl.load` and `tl.store` lines carefully.

2. **The fp16 softmax NaN.** A softmax kernel that worked in fp32 NaNs when you switch the input to fp16. The kernel computes `e = tl.exp(x_row)` and then divides by `tl.sum(e)`. Add the missing line that prevents overflow.

   *Hint:* max-subtract trick.

3. **TRITON_INTERPRET ground truth.** Take any kernel from a previous chapter, set `TRITON_INTERPRET=1`, add `tl.device_print("offs", offs)` and `tl.device_print("mask", mask)` to one program, and confirm with your eyes that the offsets are what you expect for shape `N=BLOCK+5`. Submit your printed output as a screenshot/log.

   *Hint:* `os.environ["TRITON_INTERPRET"] = "1"` before importing triton.

## Intermediate

4. **The stride bug.** A row-wise sum kernel works for contiguous inputs and produces garbage for the transposed view `x.t()`. The kernel uses `x_ptr + pid * N + offs`. Rewrite it to use `pid * stride_m + offs * stride_n` and accept stride arguments. Verify on both contiguous and transposed inputs.

   *Hint:* never bake stride assumptions into the offset math; always pass strides from the launcher.

5. **The dtype bug.** A matmul kernel correctly computes `Y = X @ W` in fp32 but produces visibly wrong results in fp16. The accumulator is `tl.zeros((BLOCK_M, BLOCK_N), dtype=x.dtype)`. Fix it without changing input or output dtypes.

   *Hint:* the chapter's golden rule about accumulator dtype.

## Advanced

6. **The triple bug.** The following kernel has three bugs (mask, stride, dtype). Find all three. Write a test that catches each one individually (i.e. fix two bugs, leave one, watch the test for that bug fail).

   ```python
   @triton.jit
   def buggy_sum_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offs = tl.arange(0, BLOCK)
       x = tl.load(x_ptr + pid * N + offs)              # bug 1: no mask
       acc = tl.zeros((), dtype=tl.float16)             # bug 2: fp16 acc
       acc += tl.sum(x, axis=0)
       tl.store(out_ptr + pid, acc)
       # bug 3 lives in the launcher — it uses x.stride() but assumes
       # a particular layout. See solutions for the actual launcher.
   ```

   *Hint:* go one bug at a time. After each fix, re-run the test.
