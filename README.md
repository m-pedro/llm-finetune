# llm-finetune

A Parameter-Efficient Fine-Tuning (`PEFT`) and Post-Training Alignment library built from scratch in Python and PyTorch. Implements **LoRA**, **DoRA** (Weight-Decomposed Low-Rank Adaptation), **4-bit NormalFloat (`NF4`) QLoRA** quantization with nibble packing, **Supervised Fine-Tuning (`SFT`)** with prompt-loss masking, and **Direct Preference Optimization (`DPO`)** alignment.

## 15-Day Incremental Engineering Roadmap (Month 4, Days 106–120)

- [x] **Day 106:** Initialize `llm_finetune` package, `LoRAConfig`, MIT license, and 15-day roadmap
- [ ] **Day 107:** Implement `LoRALinear` with Kaiming-A / zero-B initialization and weight merging (`llm_finetune/lora.py`)
- [ ] **Day 108:** Implement recursive target-module LoRA injection, freezing, and adapter state dicts (`llm_finetune/inject.py`)
- [ ] **Day 109:** Implement `DoRALinear` decomposing weights into magnitude and directional updates (`llm_finetune/dora.py`)
- [ ] **Day 110:** Implement 4-bit NormalFloat (`NF4`) blockwise quantizer and `QLoRALinear` (`llm_finetune/qlora.py`)
- [ ] **Day 111:** Implement ChatML instruction formatter with prompt-token loss masking (`llm_finetune/chat.py`)
- [ ] **Day 112:** Build Supervised Fine-Tuning (`SFT`) dataset and dynamic padding collator (`llm_finetune/sft_data.py`)
- [ ] **Day 113:** Implement `SFTTrainer` with gradient accumulation and adapter checkpointing (`llm_finetune/sft_trainer.py`)
- [ ] **Day 114:** Implement Direct Preference Optimization (`DPO`) loss and reward margin metrics (`llm_finetune/dpo_loss.py`)
- [ ] **Day 115:** Build `DPOTrainer` aligning policy LoRA model against frozen reference model (`llm_finetune/dpo_trainer.py`)
- [ ] **Day 116:** Add standalone causal Transformer backbone for end-to-end SFT & DPO alignment (`llm_finetune/backbone.py`)
- [ ] **Day 117:** Add end-to-end SFT instruction following and DPO preference margin convergence tests
- [ ] **Day 118:** Build adapter merge-and-export utility verifying bit-exact forward parity
- [ ] **Day 119:** Build `llm-finetune` CLI and Full-FT vs LoRA vs QLoRA memory/throughput profiler
- [ ] **Day 120:** Complete PEFT & alignment architecture guide, math reference table, and benchmarks
