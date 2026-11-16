import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA, Liu et al., ICML 2024)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.scaling = lora_alpha / float(r)

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Learnable magnitude vector m initialized to row-wise L2 norm of W_0: shape (out_features, 1)
        init_mag = base_layer.weight.data.norm(p=2, dim=1, keepdim=True)
        self.dora_magnitude = nn.Parameter(init_mag.clone())

        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def effective_weight(self) -> torch.Tensor:
        """Computes m * (W_0 + s * B @ A) / ||W_0 + s * B @ A||_2."""
        combined = self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling
        col_norm = combined.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        direction = combined / col_norm
        return self.dora_magnitude * direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dropped = self.dropout(x)
        w = self.effective_weight()
        return F.linear(x_dropped, w, self.base_layer.bias)
