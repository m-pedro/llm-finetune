from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

IGNORE_INDEX = -100


@dataclass
class ChatExample:
    user: str
    assistant: str
    system: str = "You are a helpful AI assistant."


def format_chatml_segments(example: ChatExample) -> Tuple[str, str]:
    """Returns (prompt_prefix, assistant_completion) formatted in ChatML."""
    prompt_prefix = (
        f"<|im_start|>system\n{example.system}<|im_end|>\n"
        f"<|im_start|>user\n{example.user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    completion = f"{example.assistant}<|im_end|>\n"
    return prompt_prefix, completion


def encode_sft_example(
    example: ChatExample,
    encode_fn: Callable[[str], List[int]],
    max_seq_len: int = 256,
) -> Dict[str, List[int]]:
    """Encodes a ChatExample into causal (input_ids, labels) with prompt tokens masked to -100."""
    prompt_text, completion_text = format_chatml_segments(example)
    prompt_ids = encode_fn(prompt_text)
    completion_ids = encode_fn(completion_text)

    full_ids = (prompt_ids + completion_ids)[: max_seq_len + 1]
    if len(full_ids) < 2:
        raise ValueError("encoded conversation must contain at least 2 tokens")

    # Next-token shift: input_ids[t] predicts full_ids[t + 1]
    input_ids = full_ids[:-1]
    raw_targets = full_ids[1:]

    # Prompt tokens reside at indices 0 .. len(prompt_ids) - 1 in full_ids.
    # In raw_targets (which is shifted left by 1), positions < len(prompt_ids) - 1
    # predict prompt tokens and MUST be masked with IGNORE_INDEX (-100).
    prompt_target_cutoff = max(0, len(prompt_ids) - 1)
    labels = [
        IGNORE_INDEX if idx < prompt_target_cutoff else tok
        for idx, tok in enumerate(raw_targets)
    ]

    return {"input_ids": input_ids, "labels": labels}


def byte_encode(text: str) -> List[int]:
    """Deterministic UTF-8 byte tokenizer (offset by +4 to reserve 0..3 for special tokens)."""
    return [b + 4 for b in text.encode("utf-8")]
