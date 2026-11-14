import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper around a frozen nn.Linear layer (Hu et al., 2021)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / math.sqrt(r)) if use_rslora else (lora_alpha / float(r))
        self.merged = False

        # Freeze pre-trained base weights and bias
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # A: (r, in_features) initialized with Kaiming uniform; B: (out_features, r) initialized to 0
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def delta_weight(self) -> torch.Tensor:
        """Computes the low-rank weight update (alpha / r) * (B @ A) of shape (out_features, in_features)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """Folds (alpha / r) * B @ A into base_layer.weight for zero-overhead inference."""
        if not self.merged:
            self.base_layer.weight.data.add_(self.delta_weight())
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Subtracts (alpha / r) * B @ A from base_layer.weight to restore separate adapter mode."""
        if self.merged:
            self.base_layer.weight.data.sub_(self.delta_weight())
            self.merged = False
