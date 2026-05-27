"""Chapter 23 — Int8 / FP8 style dequant and matmul.

Exports:
    dequant_int8_per_channel
    dequant_bias_gelu_fused
    w8a16_matmul
"""

from .dequant import dequant_int8_per_channel
from .dequant_bias_gelu import dequant_bias_gelu_fused
from .w8a16_matmul import w8a16_matmul

__all__ = [
    "dequant_int8_per_channel",
    "dequant_bias_gelu_fused",
    "w8a16_matmul",
]
