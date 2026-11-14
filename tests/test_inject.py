import torch
import torch.nn as nn
from llm_finetune.config import LoRAConfig
from llm_finetune.inject import (
    count_trainable_parameters,
    get_lora_state_dict,
    inject_lora,
    load_lora_state_dict,
)
from llm_finetune.lora import LoRALinear


class DummyAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(64, 64, bias=False)
        self.wk = nn.Linear(64, 64, bias=False)
        self.wv = nn.Linear(64, 64, bias=False)
        self.untargeted = nn.Linear(64, 64, bias=False)


def test_inject_lora_freezes_base_and_saves_compact_adapter_state():
    torch.manual_seed(7)
    model = DummyAttentionBlock()
    cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=("wq", "wv"))
    inject_lora(model, cfg)

    assert isinstance(model.wq, LoRALinear)
    assert isinstance(model.wv, LoRALinear)
    assert isinstance(model.wk, nn.Linear)
    assert isinstance(model.untargeted, nn.Linear)

    stats = count_trainable_parameters(model)
    # Only wq and wv have LoRA (each has 2 * 4 * 64 = 512 trainable params -> 1024 total)
    assert stats["trainable_params"] == 1024
    assert stats["trainable_percentage"] < 10.0

    adapter_sd = get_lora_state_dict(model)
    assert len(adapter_sd) == 4  # wq.lora_A, wq.lora_B, wv.lora_A, wv.lora_B

    for tensor in adapter_sd.values():
        tensor.fill_(0.25)
    load_lora_state_dict(model, adapter_sd)
    assert torch.allclose(model.wq.lora_B, torch.full_like(model.wq.lora_B, 0.25))
