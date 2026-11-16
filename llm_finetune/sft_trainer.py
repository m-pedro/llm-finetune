from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm_finetune.chat import IGNORE_INDEX
from llm_finetune.inject import get_lora_state_dict, load_lora_state_dict
from llm_finetune.sft_data import sft_collate_fn


@dataclass
class SFTConfig:
    lr: float = 3e-3
    weight_decay: float = 0.01
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_steps: int = 50
    grad_clip: float = 1.0


class SFTTrainer:
    """Supervised Fine-Tuning trainer updating only trainable LoRA/DoRA adapter parameters."""

    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("model has no trainable parameters; did you call inject_lora()?")

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.lr, weight_decay=config.weight_decay
        )

    def train(self, dataset: Dataset) -> List[float]:
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=sft_collate_fn,
        )
        self.model.train()
        losses: List[float] = []
        step = 0
        micro_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        while step < self.cfg.max_steps:
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=IGNORE_INDEX,
                )
                scaled_loss = loss / float(self.cfg.grad_accum_steps)
                scaled_loss.backward()
                micro_step += 1

                if micro_step % self.cfg.grad_accum_steps == 0:
                    if self.cfg.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.cfg.grad_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    losses.append(float(loss.item()))
                    step += 1
                    if step >= self.cfg.max_steps:
                        break

        return losses

    def save_adapter(self, path: str | Path) -> None:
        torch.save(get_lora_state_dict(self.model), Path(path))

    def load_adapter(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        load_lora_state_dict(self.model, state)
