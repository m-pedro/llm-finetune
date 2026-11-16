import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from llm_finetune.chat import ChatExample, byte_encode, encode_sft_example
from llm_finetune.dpo_loss import dpo_loss, sequence_log_probs
from llm_finetune.sft_data import sft_collate_fn


@dataclass
class PreferenceExample:
    user: str
    chosen: str
    rejected: str
    system: str = "You are a helpful AI assistant."


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen_response, rejected_response) preference triples for DPO."""

    def __init__(
        self,
        examples: List[PreferenceExample],
        encode_fn: Callable[[str], List[int]] = byte_encode,
        max_seq_len: int = 256,
    ) -> None:
        self.items = []
        for ex in examples:
            chosen_enc = encode_sft_example(
                ChatExample(user=ex.user, assistant=ex.chosen, system=ex.system),
                encode_fn=encode_fn,
                max_seq_len=max_seq_len,
            )
            rejected_enc = encode_sft_example(
                ChatExample(user=ex.user, assistant=ex.rejected, system=ex.system),
                encode_fn=encode_fn,
                max_seq_len=max_seq_len,
            )
            self.items.append({"chosen": chosen_enc, "rejected": rejected_enc})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, List[int]]]:
        return self.items[idx]


def dpo_collate_fn(
    batch: List[Dict[str, Dict[str, List[int]]]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    chosen_batch = sft_collate_fn([b["chosen"] for b in batch], pad_token_id=pad_token_id)
    rejected_batch = sft_collate_fn([b["rejected"] for b in batch], pad_token_id=pad_token_id)
    return {
        "chosen_input_ids": chosen_batch["input_ids"],
        "chosen_labels": chosen_batch["labels"],
        "rejected_input_ids": rejected_batch["input_ids"],
        "rejected_labels": rejected_batch["labels"],
    }


@dataclass
class DPOConfig:
    beta: float = 0.2
    lr: float = 5e-3
    weight_decay: float = 0.01
    batch_size: int = 4
    max_steps: int = 40
    grad_clip: float = 1.0
    label_smoothing: float = 0.0


class DPOTrainer:
    """Direct Preference Optimization trainer aligning a LoRA policy against a frozen reference."""

    def __init__(
        self,
        policy_model: nn.Module,
        config: DPOConfig,
        ref_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.policy = policy_model
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.policy.to(self.device)

        if ref_model is None:
            self.ref = copy.deepcopy(policy_model).to(self.device)
        else:
            self.ref = ref_model.to(self.device)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("policy_model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.lr, weight_decay=config.weight_decay
        )

    def _forward_logps(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        return sequence_log_probs(logits, labels)

    def train(self, dataset: Dataset) -> List[Dict[str, float]]:
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=dpo_collate_fn,
        )
        self.policy.train()
        history: List[Dict[str, float]] = []
        step = 0

        while step < self.cfg.max_steps:
            for batch in loader:
                if step >= self.cfg.max_steps:
                    break
                c_ids = batch["chosen_input_ids"].to(self.device)
                c_lbl = batch["chosen_labels"].to(self.device)
                r_ids = batch["rejected_input_ids"].to(self.device)
                r_lbl = batch["rejected_labels"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                pol_c_logps = self._forward_logps(self.policy, c_ids, c_lbl)
                pol_r_logps = self._forward_logps(self.policy, r_ids, r_lbl)

                with torch.no_grad():
                    ref_c_logps = self._forward_logps(self.ref, c_ids, c_lbl)
                    ref_r_logps = self._forward_logps(self.ref, r_ids, r_lbl)

                loss, metrics = dpo_loss(
                    pol_c_logps,
                    pol_r_logps,
                    ref_c_logps,
                    ref_r_logps,
                    beta=self.cfg.beta,
                    label_smoothing=self.cfg.label_smoothing,
                )
                loss.backward()

                if self.cfg.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.policy.parameters() if p.requires_grad],
                        self.cfg.grad_clip,
                    )
                self.optimizer.step()
                history.append(metrics)
                step += 1

        return history
