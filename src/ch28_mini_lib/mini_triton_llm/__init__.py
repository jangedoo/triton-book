"""mini_triton_llm: a tiny consolidated kernel library.

Re-exports the public API for every kernel we built in Parts 2 through 5,
plus the benchmarking and testing helpers from Part 7. Importing this
package alone is enough to wire a transformer block's hot path through
Triton kernels.

Public API:

    from mini_triton_llm import (
        rmsnorm,
        residual_rmsnorm,
        softmax,
        rope,
        swiglu,
        flash_attention,
        cross_entropy,
        benchmarking,
        testing,
    )
"""

from . import benchmarking, testing
from .attention import flash_attention
from .cross_entropy import cross_entropy
from .residual_rmsnorm import residual_rmsnorm
from .rmsnorm import rmsnorm
from .rope import rope
from .softmax import softmax
from .swiglu import swiglu

__all__ = [
    "rmsnorm",
    "residual_rmsnorm",
    "softmax",
    "rope",
    "swiglu",
    "flash_attention",
    "cross_entropy",
    "benchmarking",
    "testing",
]

__version__ = "0.1.0"
