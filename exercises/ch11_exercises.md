# Chapter 11 Exercises

These are lighter than other chapters. Most of Chapter 11 is conceptual
because the kernels need hardware the author cannot run.

## Beginner

1. **Why persistent + grouped ordering combine well.** Write a short
   paragraph (3-5 sentences) in your own words. Hint: persistent keeps
   the same SM hot across many tiles; grouped ordering ensures those
   tiles share rows of A.

2. **Read the Triton persistent matmul tutorial source.** Find the
   official tutorial at `python/tutorials/09-persistent-matmul.py` in
   the Triton repo. Identify the outer program-loop (the `for tile_id
   in range(start_pid, num_tiles, NUM_SMS):` line in our skeleton).
   What does the tutorial do that our skeleton does not?

3. **Estimate launch-overhead savings.** A 4096x4096 matmul with
   128x128 tiles has 32 * 32 = 1024 output tiles. On a hypothetical
   84-SM GPU, the regular kernel launches 1024 programs and the
   persistent kernel launches 84. If launch overhead is roughly 5 us
   per program, what fraction of the total kernel time would that
   overhead represent for a regular vs persistent kernel that takes
   2 ms total?

## Intermediate

4. **Rewrite Ch 9 grouped matmul as persistent.** Take
   `src/ch09_matmul/grouped_matmul.py` and add the outer for-loop over
   `tile_id` so each program owns multiple tiles. Confirm the result
   matches the non-persistent version on a small shape. (You can run
   this on sm_75; performance will not improve but correctness is the
   point.)

5. **Sketch FP8 kernel signature.** What new arguments does an FP8
   matmul kernel need that an fp16 one does not? Sketch the function
   signature including the per-block scale tensors and explain where
   the scales get applied (hint: just before the fp32 accumulate).

## Advanced

6. **Block-scaled matmul tutorial summary.** Read the official Triton
   block-scaled matmul tutorial. In 1-2 paragraphs, summarize: (a) what
   the scaling-tensor layout looks like, (b) how the kernel interleaves
   the scale loads with the data loads, (c) why per-block scaling beats
   per-tensor scaling for activations with outliers.

Solutions in `solutions/ch11_solutions.py` (prose for conceptual items).
