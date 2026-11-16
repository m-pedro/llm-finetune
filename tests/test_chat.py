from llm_finetune.chat import (
    IGNORE_INDEX,
    ChatExample,
    byte_encode,
    encode_sft_example,
    format_chatml_segments,
)


def test_chatml_prompt_masking_only_supervises_assistant_tokens():
    ex = ChatExample(
        system="Be concise.",
        user="2+2?",
        assistant="4",
    )
    prompt_str, comp_str = format_chatml_segments(ex)
    encoded = encode_sft_example(ex, byte_encode, max_seq_len=512)

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    assert len(input_ids) == len(labels)

    prompt_len = len(byte_encode(prompt_str))
    comp_ids = byte_encode(comp_str)

    # All target positions predicting prompt tokens are masked with -100
    assert all(lbl == IGNORE_INDEX for lbl in labels[: prompt_len - 1])
    # All target positions predicting the assistant completion match comp_ids exactly
    assert labels[prompt_len - 1 :] == comp_ids
