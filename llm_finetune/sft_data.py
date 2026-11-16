from typing import Callable, Dict, List
import torch
from torch.utils.data import Dataset

from llm_finetune.chat import IGNORE_INDEX, ChatExample, byte_encode, encode_sft_example


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset storing prompt-masked (input_ids, labels) pairs."""

    def __init__(
        self,
        examples: List[ChatExample],
        encode_fn: Callable[[str], List[int]] = byte_encode,
        max_seq_len: int = 256,
    ) -> None:
        self.records = [
            encode_sft_example(ex, encode_fn=encode_fn, max_seq_len=max_seq_len)
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.records[idx]


def sft_collate_fn(
    batch: List[Dict[str, List[int]]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Pads variable-length SFT sequences to the longest sequence in the batch."""
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
