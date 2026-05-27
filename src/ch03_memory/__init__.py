"""Chapter 3: memory, pointers, strides, and masks.

Exports three kernels that together cover the access patterns you will see
for the rest of the book:

- ``copy``: the simplest 1-D pattern.
- ``row_add``: a 2-D pattern with one program per row.
- ``transpose``: a 2-D tile-read / tile-write pattern with non-contiguous
  output strides.
"""

from .copy import copy, copy_kernel
from .row_add import row_add, row_add_kernel
from .transpose import transpose, transpose_kernel

__all__ = [
    "copy", "copy_kernel",
    "row_add", "row_add_kernel",
    "transpose", "transpose_kernel",
]
