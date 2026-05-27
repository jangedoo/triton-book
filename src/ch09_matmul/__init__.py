"""Chapter 9: matmul from scratch.

Exports the naive row-major matmul launcher and the grouped-ordering matmul
launcher. Both compute C = A @ B where A is (M, K) and B is (K, N).
"""

from .naive_matmul import matmul_naive, matmul_naive_kernel
from .grouped_matmul import matmul_grouped, matmul_grouped_kernel

__all__ = [
    "matmul_naive",
    "matmul_naive_kernel",
    "matmul_grouped",
    "matmul_grouped_kernel",
]
