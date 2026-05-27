"""Chapter 29: drop the Triton kernels from `mini_triton_llm` into a real
PyTorch transformer block and compare correctness, latency, and memory.
"""

from .pytorch_block import PyTorchBlock, BlockConfig
from .triton_block import TritonBlock

__all__ = ["PyTorchBlock", "TritonBlock", "BlockConfig"]
