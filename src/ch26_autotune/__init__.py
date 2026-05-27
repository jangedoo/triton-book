"""Chapter 26 — autotuning kernels."""

from .autotuned_softmax import autotuned_softmax
from .autotuned_matmul import autotuned_matmul

__all__ = ["autotuned_softmax", "autotuned_matmul"]
