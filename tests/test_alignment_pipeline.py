import torch
from llm_finetune.backbone import BackboneConfig, CausalLMBackbone
from llm_finetune.chat import ChatExample
from llm_finetune.config import LoRAConfig
from llm_finetune.dpo_trainer import DPOConfig, DPOTrainer, PreferenceDataset, PreferenceExample
from llm_finetune.inject import inject_lora
from llm_finetune.sft_data import SFTDataset
from llm_finetune.sft_trainer import SFTConfig, SFTTrainer


def test_two_stage_sft_then_dpo_alignment_pipeline():
    torch.manual_seed(42)
    bcfg = BackboneConfig(vocab_size=264, dim=64, n_layers=2, n_heads=4, hidden_dim=128)
    model = CausalLMBackbone(bcfg)
    inject_lora(model, LoRAConfig(r=8, lora_alpha=16.0))

    # Stage 1: Supervised Fine-Tuning (SFT) on instruction-response pairs
    sft_examples = [
        ChatExample(user="Status?", assistant="OK: healthy"),
        ChatExample(user="Memory?", assistant="OK: 4-bit NF4"),
    ] * 6
    sft_ds = SFTDataset(sft_examples, max_seq_len=128)
    sft_trainer = SFTTrainer(
        model, SFTConfig(lr=5e-3, batch_size=4, grad_accum_steps=1, max_steps=25)
    )
    sft_losses = sft_trainer.train(sft_ds)
    assert sft_losses[-1] < sft_losses[0]

    # Stage 2: Direct Preference Optimization (DPO) aligning the SFT model toward chosen answers
    dpo_examples = [
        PreferenceExample(
            user="Status?",
            chosen="OK: healthy and verified",
            rejected="ERR: unknown crash",
        ),
        PreferenceExample(
            user="Memory?",
            chosen="OK: 4-bit NF4 quantized",
            rejected="ERR: out of memory",
        ),
    ] * 6
    dpo_ds = PreferenceDataset(dpo_examples, max_seq_len=128)
    dpo_trainer = DPOTrainer(
        model, DPOConfig(beta=0.25, lr=5e-3, batch_size=4, max_steps=25)
    )
    dpo_history = dpo_trainer.train(dpo_ds)

    assert dpo_history[-1]["loss"] < dpo_history[0]["loss"]
    assert dpo_history[-1]["reward_margin"] > 0.5
    assert dpo_history[-1]["reward_accuracy"] == 1.0
