"""Chapter 11: persistent matmul skeleton.

The persistent kernel here is intentionally untested -- the author's
target hardware (2070 SUPER, sm_75) does not have the tensor-core
generations that make persistent matmul a clear win. The code is provided
so readers with sm_80+ hardware can try it. See module docstrings for
hardware requirements.
"""

from .persistent_matmul_skeleton import (
    persistent_matmul_skeleton,
    persistent_matmul_kernel,
)

__all__ = ["persistent_matmul_skeleton", "persistent_matmul_kernel"]
