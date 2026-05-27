"""Chapter 30: the worked example kernel used to walk the playbook —
fused `residual + LayerNorm + dropout`.
"""

from .fused_residual_ln_dropout import fused_residual_ln_dropout

__all__ = ["fused_residual_ln_dropout"]
