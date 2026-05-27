# Chapter 30 Exercises

These are not "implement a kernel" exercises. They are "design a kernel
from scratch using the playbook" exercises. Pick a layer, walk the
fifteen steps, write the code.

## Beginner

### Exercise 1: Pick a layer to fuse

Pick *one* of: `Linear + ReLU`, `Linear + GELU`, `bias + dropout`. Write
out steps 1 through 3 of the playbook (PyTorch reference, inputs/outputs
with shapes, reduction-axis analysis) in `solutions/ch30_solutions.py`
under a heading for your chosen layer. Do not implement the kernel.

### Exercise 2: Identify shapes for grouped-query attention

Write the input/output shape table for grouped-query attention with
`H_q=8` query heads and `H_kv=2` key/value heads (each KV head serves
four query heads). What changes versus standard multi-head attention?
Document this as a comment block in `solutions/ch30_solutions.py`.

### Exercise 3: Sketch a sliding-window attention kernel

Re-use the FlashAttention forward kernel design (Chapter 14) and modify
the inner loop so each query position `i` only attends to keys in the
window `[max(0, i - W), i]` for a fixed `W=128`. Walk through steps 4
through 7 of the playbook (block mapping, loads, computes, stores) in
prose. Do not implement.

## Intermediate

### Exercise 4: Implement your Beginner-1 kernel

Take the design from Exercise 1 and write the kernel. Walk steps 8
through 12 (masks, dtype policy, tiny-shape test, weird-shape test,
benchmark). Commit the Triton kernel as `src/ch30_playbook/your_op.py`
and the tests as `tests/test_ch30_your_op.py`.

### Exercise 5: Implement your sliding-window attention

Take the sketch from Exercise 3 and implement it. The inner loop will
look almost identical to `attention.py` from the mini library; the only
change is the `start_n` lower bound and the loop trip count. Test it
against a PyTorch reference that explicitly masks the disallowed
positions. Document the masking edge cases (start of sequence, partial
window blocks) in `solutions/ch30_solutions.py`.

## Advanced

### Exercise 6: Implement, benchmark, and autotune your Beginner-1 kernel

Take the kernel from Exercise 4 to the optimization phase. Wrap it in
`triton.autotune` with at least four `(BLOCK_SIZE, num_warps,
num_stages)` configs that span the realistic shape space. Add a
roofline analysis: is the kernel memory-bound or compute-bound at the
shapes you care about? Document the autotune config you would ship and
why. Write the results into the Chapter 30 benchmark notebook so the
table on the chapter page reflects your numbers.
