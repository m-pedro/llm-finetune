import torch
import torch.nn as nn
from llm_finetune.qlora import QLoRALinear


def test_nf4_qlora_memory_compression_and_adapter_backprop():
    torch.manual_seed(123)
    base = nn.Linear(128, 128, bias=False)
    raw_fp32_bytes = base.weight.numel() * 4  # 65,536 bytes

    qlora = QLoRALinear(base, r=8, lora_alpha=16.0, block_size=64)
    compressed_bytes = qlora.base_memory_bytes()

    # 16,384 weights -> 8,192 packed uint8 bytes + 256 fp32 scales (1,024 bytes) = 9,216 bytes (>7x smaller!)
    assert compressed_bytes == 8192 + 1024
    assert raw_fp32_bytes / compressed_bytes > 7.0

    # Verify dequantization accuracy on Gaussian weights
    w_hat = qlora.dequantized_base_weight()
    rel_err = (base.weight.data - w_hat).norm() / base.weight.data.norm()
    assert rel_err < 0.12

    # Verify forward + backward updates lora_A and lora_B
    x = torch.randn(2, 6, 128, requires_grad=True)
    y = qlora(x)
    y.sum().backward()
    assert qlora.lora_B.grad is not None
    assert qlora.lora_A.grad is not None
