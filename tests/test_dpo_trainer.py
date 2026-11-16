import torch
from llm_finetune.config import LoRAConfig
from llm_finetune.dpo_trainer import DPOConfig, DPOTrainer, PreferenceDataset, PreferenceExample
from llm_finetune.inject import inject_lora
from tests.test_sft_trainer import TinyCausalLM


def test_dpo_trainer_increases_chosen_over_rejected_reward_margin():
    torch.manual_seed(42)
    model = TinyCausalLM()
    inject_lora(model, LoRAConfig(r=4, lora_alpha=8.0, target_modules=("wq", "wv")))

    prefs = [
        PreferenceExample(
            user="Write safe Rust.",
            chosen="Use ownership and borrowing.",
            rejected="Deref null raw pointers everywhere.",
        )
    ] * 8
    ds = PreferenceDataset(prefs, max_seq_len=128)
    trainer = DPOTrainer(model, DPOConfig(beta=0.2, lr=1e-2, batch_size=4, max_steps=20))
    history = trainer.train(ds)

    # At step 0, policy == ref so reward_margin == 0.0; after DPO training, margin > 0 and accuracy == 1.0
    assert history[0]["reward_margin"] == 0.0
    assert history[-1]["reward_margin"] > 0.5
    assert history[-1]["reward_accuracy"] == 1.0
    assert history[-1]["loss"] < history[0]["loss"]
