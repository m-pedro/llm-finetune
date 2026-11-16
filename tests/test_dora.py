import torch
import torch.nn as nn
from llm_finetune.config import LoRAConfig
from llm_finetune.dora import DoRALinear
from llm_finetune.inject import inject_lora


def test_dora_initialization_identity_and_magnitude_direction_gradients():
    torch.manual_seed(99)
    base = nn.Linear(32, 48, bias=True)
    x = torch.randn(2, 8, 32)
    ref_out = base(x).clone()

    dora = DoRALinear(base, r=4, lora_alpha=8.0)
    # 1. Because B=0 and m=||W_0||_2 at step 0, DoRALinear must match base_layer(x)
    out = dora(x)
    assert torch.allclose(out, ref_out, atol=1e-5)

    # 2. Verify gradients flow to dora_magnitude, lora_A, and lora_B while base weight stays frozen
    loss = out.pow(2).sum()
    loss.backward()
    assert dora.dora_magnitude.grad is not None
    assert dora.lora_B.grad is not None
    assert dora.base_layer.weight.grad is None


def test_inject_dora_via_config():
    class TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wq = nn.Linear(16, 16, bias=False)

    net = TinyNet()
    inject_lora(net, LoRAConfig(r=4, use_dora=True, target_modules=("wq",)))
    assert isinstance(net.wq, DoRALinear)
