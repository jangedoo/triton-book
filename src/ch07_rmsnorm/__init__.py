"""Chapter 7: RMSNorm forward (and backward).

Public API:
    rmsnorm(x, weight, eps): forward, matches the LLaMA/Mistral/Gemma formulation.
"""

from .rmsnorm import rmsnorm, rmsnorm_backward

__all__ = ["rmsnorm", "rmsnorm_backward"]
