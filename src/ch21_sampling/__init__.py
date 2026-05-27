from .temperature import temperature_scale
from .argmax import argmax_sample
from .top_k import top_k_mask, top_p_mask

__all__ = [
    "temperature_scale",
    "argmax_sample",
    "top_k_mask",
    "top_p_mask",
]
