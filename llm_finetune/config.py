from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = field(
        default_factory=lambda: ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
    )
    use_dora: bool = False
    use_rslora: bool = False
    nf4_block_size: int = 64

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {self.r}")
        if self.lora_alpha <= 0.0:
            raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.nf4_block_size <= 0 or (self.nf4_block_size & (self.nf4_block_size - 1)) != 0:
            raise ValueError(f"nf4_block_size must be a power of 2, got {self.nf4_block_size}")

    @property
    def scaling(self) -> float:
        """Computes standard LoRA scaling (alpha / r) or Rank-Stabilized rsLoRA scaling (alpha / sqrt(r))."""
        if self.use_rslora:
            return self.lora_alpha / (self.r ** 0.5)
        return self.lora_alpha / float(self.r)
