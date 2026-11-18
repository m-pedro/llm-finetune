import torch
import torch.nn as nn

from llm_finetune.dora import DoRALinear
from llm_finetune.lora import LoRALinear


@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """Folds all LoRALinear and DoRALinear weights into standard nn.Linear layers in-place."""
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            child.merge()
            setattr(model, name, child.base_layer)
        elif isinstance(child, DoRALinear):
            merged_w = child.effective_weight()
            child.base_layer.weight.data.copy_(merged_w)
            setattr(model, name, child.base_layer)
        else:
            merge_and_unload(child)
    return model
