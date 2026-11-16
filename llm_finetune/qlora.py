import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Exact 16-entry NormalFloat4 (NF4) quantization codebook in [-1.0, 1.0] (Dettmers et al., 2023)
NF4_CODEBOOK = torch.tensor(
    [
        -1.00000000,
        -0.69619280,
        -0.52507305,
        -0.39491749,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.00000000,
        0.07958030,
        0.16093020,
        0.24611230,
        0.33791524,
        0.44070983,
        0.56261700,
        0.72295684,
        1.00000000,
    ],
    dtype=torch.float32,
)


def quantize_nf4(
    weight: torch.Tensor, block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantizes a 2D tensor into packed uint8 NF4 nibbles (2 values/byte) + per-block fp32 absmax."""
    orig_shape = weight.shape
    flat = weight.detach().float().reshape(-1)
    if flat.numel() % block_size != 0:
        raise ValueError(
            f"weight numel ({flat.numel()}) must be divisible by block_size ({block_size})"
        )

    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = (blocks / absmax).clamp(-1.0, 1.0)

    codebook = NF4_CODEBOOK.to(weight.device)
    # Find nearest NF4 codebook index in [0..15] for each element
    diffs = (normalized.unsqueeze(-1) - codebook).abs()
    indices = torch.argmin(diffs, dim=-1).to(torch.uint8).reshape(-1)

    # Pack two 4-bit indices into each uint8 byte: high nibble = even index, low nibble = odd index
    high = indices[0::2] << 4
    low = indices[1::2] & 0x0F
    packed = (high | low).contiguous()
    return packed, absmax.squeeze(1).contiguous(), orig_shape


def dequantize_nf4(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    orig_shape: torch.Size,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpacks uint8 NF4 nibbles and rescales by per-block absmax back to `orig_shape`."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    indices = torch.stack((high, low), dim=-1).reshape(-1).long()

    codebook = NF4_CODEBOOK.to(packed.device, dtype=dtype)
    dequant_blocks = codebook[indices].view(-1, block_size)
    rescaled = dequant_blocks * absmax.to(dtype).unsqueeze(1)
    return rescaled.view(orig_shape)


class QLoRALinear(nn.Module):
    """4-bit NormalFloat (NF4) Quantized Linear layer with high-precision LoRA adapters."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.scaling = lora_alpha / float(r)
        self.block_size = block_size
        self.weight_shape = base_layer.weight.shape
        self.compute_dtype = base_layer.weight.dtype

        packed, absmax, _ = quantize_nf4(base_layer.weight.data, block_size=block_size)
        self.register_buffer("nf4_packed", packed)
        self.register_buffer("nf4_absmax", absmax)

        if base_layer.bias is not None:
            self.register_buffer("bias", base_layer.bias.data.detach().clone())
        else:
            self.bias = None

        device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty((r, self.in_features), device=device, dtype=self.compute_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, r), device=device, dtype=self.compute_dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def dequantized_base_weight(self) -> torch.Tensor:
        return dequantize_nf4(
            self.nf4_packed,
            self.nf4_absmax,
            self.weight_shape,
            block_size=self.block_size,
            dtype=self.compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self.dequantized_base_weight()
        base_out = F.linear(x, w_dequant, self.bias)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def base_memory_bytes(self) -> int:
        """Returns packed 4-bit buffer + absmax scale footprint in bytes."""
        return (
            self.nf4_packed.numel() * self.nf4_packed.element_size()
            + self.nf4_absmax.numel() * self.nf4_absmax.element_size()
        )
