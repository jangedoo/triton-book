"""Chapter 6: LayerNorm forward (and an optional backward).

Public API:
    layernorm(x, weight, bias, eps): forward, mirrors F.layer_norm semantics
        for `normalized_shape = (x.shape[-1],)`.
"""

from .layernorm import layernorm, layernorm_backward

__all__ = ["layernorm", "layernorm_backward"]
