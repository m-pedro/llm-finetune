import torch
import torch.nn as nn
from llm_finetune.chat import ChatExample
from llm_finetune.config import LoRAConfig
from llm_finetune.inject import inject_lora
from llm_finetune.sft_data import SFTDataset
from llm_finetune.sft_trainer import SFTConfig, SFTTrainer


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 264, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = h + torch.tanh(self.wq(h) + self.wv(h))
        return self.head(h)


def test_sft_trainer_reduces_masked_instruction_loss_and_saves_adapter(tmp_path):
    torch.manual_seed(7)
    model = TinyCausalLM()
    base_embed_snapshot = model.embed.weight.detach().clone()

    inject_lora(model, LoRAConfig(r=4, lora_alpha=8.0, target_modules=("wq", "wv")))

    ds = SFTDataset(
        [ChatExample(user="Ping", assistant="Pong")] * 8,
        max_seq_len=128,
    )
    trainer = SFTTrainer(
        model, SFTConfig(lr=1e-2, batch_size=4, grad_accum_steps=2, max_steps=20)
    )
    losses = trainer.train(ds)

    assert losses[-1] < losses[0]
    # Base embedding weights must remain strictly untouched during LoRA SFT
    assert torch.equal(model.embed.weight, base_embed_snapshot)

    adapter_path = tmp_path / "lora_adapter.pt"
    trainer.save_adapter(adapter_path)
    assert adapter_path.exists()
