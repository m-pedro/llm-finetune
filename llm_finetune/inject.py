from typing import Dict
import torch
import torch.nn as nn

from llm_finetune.config import LoRAConfig
from llm_finetune.lora import LoRALinear


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Freezes all model parameters and replaces matching Linear layers with LoRALinear."""
    for param in model.parameters():
        param.requires_grad_(False)

    _replace_Matching_linears(model, config)
    return model


def _replace_Matching_linears(parent: nn.Module, config: LoRAConfig) -> None:
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear) and any(
            name == target or name.endswith(f".{target}") for target in config.target_modules
        ):
            if config.use_dora:
                from llm_finetune.dora import DoRALinear

                wrapped: nn.Module = DoRALinear(
                    child,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )
            else:
                wrapped = LoRALinear(
                    child,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    use_rslora=config.use_rslora,
                )
            setattr(parent, name, wrapped)
        else:
            _replace_Matching_linears(child, config)


def count_trainable_parameters(model: nn.Module) -> Dict[str, int | float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": pct,
    }


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only trainable adapter parameters (lora_A, lora_B, dora_magnitude)."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    own_state = dict(model.named_parameters())
    for name, tensor in state_dict.items():
        if name not in own_state:
            raise KeyError(f"unexpected adapter parameter key: {name}")
        own_state[name].data.copy_(tensor)
