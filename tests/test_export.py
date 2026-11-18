import torch
import torch.nn as nn
from llm_finetune.backbone import BackboneConfig, CausalLMBackbone
from llm_finetune.config import LoRAConfig
from llm_finetune.dora import DoRALinear
from llm_finetune.export import merge_and_unload
from llm_finetune.inject import inject_lora
from llm_finetune.lora import LoRALinear


def test_merge_and_unload_lora_and_dora_preserves_exact_logits():
    torch.manual_seed(101)
    for use_dora in (False, True):
        cfg = BackboneConfig(vocab_size=128, dim=32, n_layers=2, n_heads=4, hidden_dim=64)
        model = CausalLMBackbone(cfg).eval()
        inject_lora(model, LoRAConfig(r=4, lora_alpha=8.0, use_dora=use_dora))

        # Perturb adapter weights away from zero to simulate fine-tuning
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (LoRALinear, DoRALinear)):
                    module.lora_B.normal_(0.0, 0.2)
                if isinstance(module, DoRALinear):
                    module.dora_magnitude.mul_(1.15)

        x = torch.randint(0, cfg.vocab_size, (2, 10))
        logits_before = model(x).clone()

        merge_and_unload(model)

        # Ensure all LoRALinear / DoRALinear wrappers have been replaced by plain nn.Linear
        assert not any(isinstance(m, (LoRALinear, DoRALinear)) for m in model.modules())
        assert isinstance(model.layers[0].attn.wq, nn.Linear)

        logits_after = model(x)
        assert torch.allclose(logits_before, logits_after, atol=1e-5)
