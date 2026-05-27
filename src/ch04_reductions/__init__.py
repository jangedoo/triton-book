"""Chapter 4: reductions.

Exports four row-wise reduction kernels that together set up the softmax and
norm chapters of Part 2.
"""

from .row_sum import row_sum, row_sum_kernel
from .row_max import row_max, row_max_kernel
from .row_mean import row_mean, row_mean_kernel
from .row_variance import row_variance, row_variance_kernel

__all__ = [
    "row_sum", "row_sum_kernel",
    "row_max", "row_max_kernel",
    "row_mean", "row_mean_kernel",
    "row_variance", "row_variance_kernel",
]
