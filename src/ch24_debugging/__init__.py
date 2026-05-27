"""Chapter 24 — Debugging Triton kernels.

This module contains a deliberately buggy kernel paired with the fixed
version, used by the chapter to demonstrate the find-and-fix workflow.
"""

from .debug_demo import (
    add_with_mask_bug,
    add_fixed,
)

__all__ = ["add_with_mask_bug", "add_fixed"]
