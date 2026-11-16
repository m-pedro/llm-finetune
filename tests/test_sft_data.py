from torch.utils.data import DataLoader
from llm_finetune.chat import IGNORE_INDEX, ChatExample
from llm_finetune.sft_data import SFTDataset, sft_collate_fn


def test_sft_dataset_and_dynamic_padding_collator():
    examples = [
        ChatExample(user="Hi", assistant="Hello!"),
        ChatExample(user="Explain LoRA in detail.", assistant="Low-Rank Adaptation freezes W0."),
    ]
    ds = SFTDataset(examples, max_seq_len=256)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=sft_collate_fn)
    batch = next(iter(loader))

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2

    # First example is shorter, so its trailing padded positions must have attention_mask == 0 and label == -100
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == IGNORE_INDEX
    assert batch["attention_mask"][1, -1].item() == 1
