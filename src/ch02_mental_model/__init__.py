"""Chapter 2: The Triton mental model.

Exports the canonical vector_add kernel and its launcher. Used by the chapter
text, the tests under tests/test_ch02_vector_add.py, and the benchmark under
benchmarks/bench_ch02_vector_add.py.
"""

from .vector_add import vector_add, vector_add_kernel

__all__ = ["vector_add", "vector_add_kernel"]
