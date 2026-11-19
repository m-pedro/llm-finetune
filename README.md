# llm-finetune

A Parameter-Efficient Fine-Tuning (`PEFT`) and Post-Training Alignment framework built from scratch in Python and PyTorch without `peft`, `bitsandbytes`, or `trl`. Implements LoRA, rsLoRA, DoRA (Weight-Decomposed Low-Rank Adaptation), 4-bit NormalFloat (`NF4`) QLoRA blockwise quantization with nibble packing, Supervised Fine-Tuning (`SFT`) with ChatML prompt-loss masking, and Direct Preference Optimization (`DPO`) alignment.

---

## Repository Structure

```text
llm-finetune/
├── LICENSE                         # MIT License
├── README.md                       # PEFT & Alignment architecture guide, math & benchmarks
├── pyproject.toml                  # Package metadata & pytest configuration
├── llm_finetune/
│   ├── __init__.py                 # Public module exports
│   ├── config.py                   # LoRAConfig (r, lora_alpha, rsLoRA, DoRA, NF4 block size)
│   ├── lora.py                     # LoRALinear with Kaiming-A / zero-B init & weight merging
│   ├── dora.py                     # DoRALinear (magnitude & directional weight decomposition)
│   ├── qlora.py                    # 4-bit NormalFloat (NF4) quantizer, uint8 packing & QLoRALinear
│   ├── inject.py                   # Recursive target-module injection & compact adapter state dicts
│   ├── chat.py                     # ChatML formatter & prompt-token loss masking (-100)
│   ├── sft_data.py                 # SFTDataset & dynamic right-padding batch collator
│   ├── sft_trainer.py              # SFTTrainer with gradient accumulation & adapter saving
│   ├── dpo_loss.py                 # Bradley-Terry DPO loss, sequence log-probs & reward margins
│   ├── dpo_trainer.py              # DPOTrainer aligning LoRA policy against frozen reference
│   ├── backbone.py                 # Self-contained Llama-style CausalLMBackbone (RMSNorm + SwiGLU)
│   ├── export.py                   # merge_and_unload() for zero-latency inference deployment
│   └── cli.py                      # CLI profiler (bench-peft, demo-sft-dpo)
└── tests/
    ├── test_config.py              # Scaling & rank validation tests
    ├── test_lora.py                # Zero-init identity & merge/unmerge parity tests
    ├── test_inject.py              # Selective module wrapping & parameter accounting tests
    ├── test_dora.py                # DoRA magnitude/direction gradient & identity tests
    ├── test_qlora.py               # 4-bit NF4 >7x compression & dequantization accuracy tests
    ├── test_chat.py                # ChatML prompt-token masking (-100) verification
    ├── test_sft_data.py            # Dynamic batch padding & attention mask tests
    ├── test_sft_trainer.py         # Masked instruction tuning & frozen base verification
    ├── test_dpo_loss.py            # DPO loss ln(2) init & chosen/rejected gradient signs
    ├── test_dpo_trainer.py         # Preference margin & accuracy convergence tests
    ├── test_backbone.py            # Causal Transformer backbone & full LoRA injection tests
    ├── test_alignment_pipeline.py  # End-to-end Pretrain -> SFT -> DPO alignment test
    └── test_export.py              # Bit-exact forward parity after merge_and_unload()
```

---

## Two-Stage Post-Training & PEFT Architecture

```text
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. Parameter-Efficient Adapter Layer (LoRA / DoRA / 4-bit QLoRA)    │
  │                                                                     │
  │              ┌───────────────────────────────┐                      │
  │              │ Frozen Base Weight W_0        │                      │
  │   x ────────►│ (fp32 / bf16 or 4-bit NF4)    ├────────► (+) ──► h   │
  │     │        └───────────────────────────────┘           ▲          │
  │     │                                                    │          │
  │     └───────► [ A: (r, d_in) ] ──► [ B: (d_out, r) ] ──► * (alpha/r)│
  │               (Kaiming Init)         (Zero Init)                    │
  └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 2. Stage 1: Supervised Fine-Tuning (SFT with Prompt Masking)        │
  │   • Input : <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n│
  │   • Labels: [ -100, -100, ..., -100,  tok_1, tok_2, ..., <|im_end|>]│
  └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 3. Stage 2: Direct Preference Optimization (DPO Alignment)          │
  │   • Maximizes margin: beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x))    │
  │                             - log(pi_theta(y_l|x)/pi_ref(y_l|x)))   │
  │   • Export: merge_and_unload() folds B @ A into W_0 (zero overhead) │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulations & Academic References

| Method | Core Equation / Mechanism | Academic Reference |
| :--- | :--- | :--- |
| **`LoRA`** | `h = W_0 x + (alpha / r) * (B @ A) x`, with `B = 0` at `t = 0` | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022) |
| **`rsLoRA`** | Rank-stabilized scaling factor `alpha / sqrt(r)` | Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (2023) |
| **`DoRA`** | `W = m * (W_0 + s * B @ A) / ||W_0 + s * B @ A||_c` | Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (ICML 2024 Oral) |
| **`QLoRA (NF4)`** | 16-level normal quantile codebook packed 2 nibbles/`uint8` + blockwise `absmax` | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023) |
| **`DPO`** | `-log sigmoid(beta * ((log pi_theta(y_w) - log pi_ref(y_w)) - (log pi_theta(y_l) - log pi_ref(y_l))))` | Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (NeurIPS 2023) |

---

## PEFT Parameter & Memory Footprint Benchmark (6-Layer, 256-D Transformer)

Run locally via `python3 -m llm_finetune.cli bench-peft` (includes model weights, buffers, `fp32` gradients, and `AdamW` first/second moment states for trainable parameters):

| Fine-Tuning Method | Trainable Parameters | % Trainable | Total Training RAM (Weights + Grads + AdamW) | Base Weight Compression |
| :--- | ---: | ---: | ---: | :--- |
| **Full Fine-Tuning (`fp32`)** | 5,049,088 | 100.00% | ~77.04 MB | 1.00x (Baseline) |
| **LoRA (`r = 8`)** | 270,336 | 5.08% | ~23.45 MB | **3.28x lower total RAM** |
| **DoRA (`r = 8`)** | 288,768 | 5.41% | ~23.73 MB | Magnitude + direction updates |
| **QLoRA `4-bit NF4` (`r = 8`)** | 270,336 | 99.32% (of float params) | **~7.30 MB** | **10.5x lower total RAM** (`>7x` base weight reduction) |

---

## Quickstart

### 1. Run All 13 Unit & Alignment Convergence Test Suites

```bash
source .venv/bin/activate
pytest -v
```

### 2. Profile Full Fine-Tuning vs. LoRA vs. DoRA vs. 4-Bit `NF4` QLoRA

```bash
python3 -m llm_finetune.cli bench-peft
```

### 3. Run End-to-End SFT -> DPO Preference Alignment & Adapter Merge Demo

```bash
python3 -m llm_finetune.cli demo-sft-dpo
```

---

## 15-Day Incremental Engineering Roadmap (Month 4, Days 106–120)

- [x] **Day 106:** Initialize `llm_finetune` package, `LoRAConfig`, MIT license, and 15-day roadmap
- [x] **Day 107:** Implement `LoRALinear` with Kaiming-A / zero-B initialization and weight merging (`llm_finetune/lora.py`)
- [x] **Day 108:** Implement recursive target-module LoRA injection, freezing, and adapter state dicts (`llm_finetune/inject.py`)
- [x] **Day 109:** Implement `DoRALinear` decomposing weights into magnitude and directional updates (`llm_finetune/dora.py`)
- [x] **Day 110:** Implement 4-bit NormalFloat (`NF4`) blockwise quantizer and `QLoRALinear` (`llm_finetune/qlora.py`)
- [x] **Day 111:** Implement ChatML instruction formatter with prompt-token loss masking (`llm_finetune/chat.py`)
- [x] **Day 112:** Build Supervised Fine-Tuning (`SFT`) dataset and dynamic padding collator (`llm_finetune/sft_data.py`)
- [x] **Day 113:** Implement `SFTTrainer` with gradient accumulation and adapter checkpointing (`llm_finetune/sft_trainer.py`)
- [x] **Day 114:** Implement Direct Preference Optimization (`DPO`) loss and reward margin metrics (`llm_finetune/dpo_loss.py`)
- [x] **Day 115:** Build `DPOTrainer` aligning policy LoRA model against frozen reference model (`llm_finetune/dpo_trainer.py`)
- [x] **Day 116:** Add standalone causal Transformer backbone for end-to-end SFT & DPO alignment (`llm_finetune/backbone.py`)
- [x] **Day 117:** Add end-to-end SFT instruction following and DPO preference margin convergence tests (`tests/test_alignment_pipeline.py`)
- [x] **Day 118:** Build adapter merge-and-export utility verifying bit-exact forward parity (`llm_finetune/export.py`)
- [x] **Day 119:** Build `llm-finetune` CLI and Full-FT vs LoRA vs QLoRA memory/throughput profiler (`llm_finetune/cli.py`)
- [x] **Day 120:** Complete PEFT & alignment architecture guide, math reference table, and benchmarks
