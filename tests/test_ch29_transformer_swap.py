"""Correctness: PyTorch block and Triton block must produce close outputs
on identical weights and identical inputs.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "ch28_mini_lib"))

from ch29_transformer_swap import BlockConfig, PyTorchBlock, TritonBlock  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")


@pytest.mark.parametrize("S", [64, 128, 512])
def test_blocks_match(S: int) -> None:
    torch.manual_seed(0)
    cfg = BlockConfig(hidden_dim=256, num_heads=4, head_dim=64, intermediate_dim=1024)

    ref = PyTorchBlock(cfg, dtype=torch.float16).cuda().eval()
    tri = TritonBlock.from_pytorch(ref).cuda().eval()

    B = 2
    x = torch.randn(B, S, cfg.hidden_dim, dtype=torch.float16, device="cuda") * 0.1

    with torch.no_grad():
        y_ref = ref(x)
        y_tri = tri(x)

    # fp16 end-to-end through SDPA vs FlashAttention; loosen slightly.
    torch.testing.assert_close(y_tri, y_ref, rtol=3e-2, atol=3e-2)


def test_triton_block_state_dict_shared() -> None:
    cfg = BlockConfig()
    ref = PyTorchBlock(cfg, dtype=torch.float16)
    tri = TritonBlock.from_pytorch(ref)
    for (kr, vr), (kt, vt) in zip(ref.state_dict().items(), tri.state_dict().items()):
        assert kr == kt
        assert torch.equal(vr, vt)
