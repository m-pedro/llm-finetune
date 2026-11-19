import copy
import sys
import time
import torch
import torch.nn as nn

from llm_finetune.backbone import BackboneConfig, CausalLMBackbone
from llm_finetune.chat import ChatExample
from llm_finetune.config import LoRAConfig
from llm_finetune.dpo_trainer import DPOConfig, DPOTrainer, PreferenceDataset, PreferenceExample
from llm_finetune.export import merge_and_unload
from llm_finetune.inject import count_trainable_parameters, inject_lora
from llm_finetune.qlora import QLoRALinear
from llm_finetune.sft_data import SFTDataset
from llm_finetune.sft_trainer import SFTConfig, SFTTrainer


def _inject_qlora(model: nn.Module, r: int = 8, alpha: float = 16.0, block_size: int = 64) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    targets = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(
                model,
                name,
                QLoRALinear(child, r=r, lora_alpha=alpha, block_size=block_size),
            )
        else:
            _inject_qlora(child, r=r, alpha=alpha, block_size=block_size)


def _model_storage_bytes(model: nn.Module) -> int:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    # Include AdamW optimizer states (m and v in fp32 = 8 bytes per trainable param) + grad (4 bytes)
    trainable_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim_and_grad_bytes = trainable_elems * 12
    return param_bytes + buf_bytes + optim_and_grad_bytes


def run_bench_peft() -> None:
    torch.manual_seed(7)
    cfg = BackboneConfig(vocab_size=512, dim=256, n_layers=6, n_heads=8, hidden_dim=768)
    base = CausalLMBackbone(cfg)

    # 1. Full Fine-Tuning
    full_model = copy.deepcopy(base)
    full_stats = count_trainable_parameters(full_model)
    full_mem_mb = _model_storage_bytes(full_model) / (1024 * 1024)

    # 2. LoRA (r=8)
    lora_model = inject_lora(copy.deepcopy(base), LoRAConfig(r=8, lora_alpha=16.0, use_dora=False))
    lora_stats = count_trainable_parameters(lora_model)
    lora_mem_mb = _model_storage_bytes(lora_model) / (1024 * 1024)

    # 3. DoRA (r=8)
    dora_model = inject_lora(copy.deepcopy(base), LoRAConfig(r=8, lora_alpha=16.0, use_dora=True))
    dora_stats = count_trainable_parameters(dora_model)
    dora_mem_mb = _model_storage_bytes(dora_model) / (1024 * 1024)

    # 4. QLoRA 4-bit NF4 (r=8)
    qlora_model = copy.deepcopy(base)
    _inject_qlora(qlora_model, r=8, alpha=16.0, block_size=64)
    qlora_stats = count_trainable_parameters(qlora_model)
    qlora_mem_mb = _model_storage_bytes(qlora_model) / (1024 * 1024)

    print("=== llm-finetune PEFT Parameter & Training Memory Profiler (6L, 256D) ===")
    print(f"{'Method':<16} | {'Trainable Params':>16} | {'% Trainable':>11} | {'Total Train RAM':>15}")
    print("-" * 68)
    for label, st, mem in [
        ("Full Fine-Tune", full_stats, full_mem_mb),
        ("LoRA (r=8)", lora_stats, lora_mem_mb),
        ("DoRA (r=8)", dora_stats, dora_mem_mb),
        ("QLoRA NF4 (r=8)", qlora_stats, qlora_mem_mb),
    ]:
        print(
            f"{label:<16} | {int(st['trainable_params']):>16,} | "
            f"{float(st['trainable_percentage']):>10.2f}% | {mem:>12.2f} MB"
        )


def run_demo_sft_dpo() -> None:
    torch.manual_seed(42)
    cfg = BackboneConfig(vocab_size=264, dim=64, n_layers=2, n_heads=4, hidden_dim=128)
    model = CausalLMBackbone(cfg)
    inject_lora(model, LoRAConfig(r=8, lora_alpha=16.0))

    sft_ds = SFTDataset(
        [
            ChatExample(user="Explain LoRA.", assistant="Low-Rank Adaptation freezes W0."),
            ChatExample(user="Explain DPO.", assistant="Direct Preference Optimization."),
        ]
        * 8,
        max_seq_len=128,
    )
    t0 = time.perf_counter()
    sft_trainer = SFTTrainer(model, SFTConfig(lr=5e-3, batch_size=4, max_steps=30))
    sft_losses = sft_trainer.train(sft_ds)

    dpo_ds = PreferenceDataset(
        [
            PreferenceExample(
                user="Explain LoRA.",
                chosen="Low-Rank Adaptation freezes W0 and trains B@A.",
                rejected="LoRA retrains every parameter in fp64.",
            )
        ]
        * 8,
        max_seq_len=128,
    )
    dpo_trainer = DPOTrainer(model, DPOConfig(beta=0.2, lr=5e-3, batch_size=4, max_steps=25))
    dpo_hist = dpo_trainer.train(dpo_ds)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    merge_and_unload(model)
    print("=== llm-finetune End-to-End SFT -> DPO Alignment Demo ===")
    print(f"SFT Loss         : {sft_losses[0]:.4f} -> {sft_losses[-1]:.4f}")
    print(f"DPO Loss         : {dpo_hist[0]['loss']:.4f} -> {dpo_hist[-1]['loss']:.4f}")
    print(f"DPO Reward Margin: {dpo_hist[0]['reward_margin']:.4f} -> {dpo_hist[-1]['reward_margin']:.4f}")
    print(f"DPO Pref Accuracy: {dpo_hist[-1]['reward_accuracy'] * 100.0:.1f}%")
    print(f"Adapter Export   : Merged & unloaded into base nn.Linear in {elapsed_ms:.1f} ms total")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "bench-peft":
        run_bench_peft()
    elif cmd == "demo-sft-dpo":
        run_demo_sft_dpo()
    else:
        print("Usage: python3 -m llm_finetune.cli <bench-peft | demo-sft-dpo>")


if __name__ == "__main__":
    main()
