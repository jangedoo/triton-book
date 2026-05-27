"""Chapter 5: Fused Softmax kernels.

Public API:
    naive_softmax(x): row-wise softmax with NO max subtraction. Overflows for fp16.
    stable_softmax(x): numerically stable row-wise softmax (single-tile rows).
    online_softmax(x): multi-tile rows via online (running-max / running-sum) reduction.
"""

from .naive_softmax import naive_softmax
from .stable_softmax import stable_softmax
from .online_softmax import online_softmax

__all__ = ["naive_softmax", "stable_softmax", "online_softmax"]
