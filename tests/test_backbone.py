import torch
from llm_finetune.backbone import BackboneConfig, CausalLMBackbone
from llm_finetune.config import LoRAConfig
from llm_finetune.inject import count_trainable_parameters, inject_lora


def test_backbone_forward_and_lora_injection_across_all_linear_projections():
    torch.manual_seed(17)
    cfg = BackboneConfig(vocab_size=264, dim=64, n_layers=2, n_heads=4, hidden_dim=128)
    model = CausalLMBackbone(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)

    inject_lora(model, LoRAConfig(r=4, lora_alpha=8.0))
    stats = count_trainable_parameters(model)
    assert stats["trainable_params"] > 0
    assert stats["trainable_percentage"] < 15.0
