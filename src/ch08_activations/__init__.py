"""Chapter 8: Activation kernels and SwiGLU.

Public API:
    gelu(x, approximate="none"|"tanh")
    silu(x)
    swiglu(x_gate, x_up)
    geglu(x_gate, x_up)
"""

from .gelu import gelu
from .silu import silu
from .swiglu import swiglu, geglu

__all__ = ["gelu", "silu", "swiglu", "geglu"]
