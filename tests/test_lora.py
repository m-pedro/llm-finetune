import torch
import torch.nn as nn
from llm_finetune.lora import LoRALinear


def test_lora_zero_init_identity_and_merge_unmerge_parity():
    torch.manual_seed(42)
    base = nn.Linear(32, 64, bias=True)
    x = torch.randn(4, 10, 32)
    expected_init = base(x).clone()

    lora = LoRALinear(base, r=4, lora_alpha=8.0)
    # 1. At step 0, B == 0 so LoRALinear output must equal base_layer(x) bit-for-bit
    assert torch.allclose(lora(x), expected_init, atol=1e-7)

    # 2. Simulate trained non-zero B weights and verify merge/unmerge parity
    with torch.no_grad():
        lora.lora_B.normal_(mean=0.0, std=0.1)

    unmerged_out = lora(x)
    lora.merge()
    assert lora.merged
    merged_out = lora(x)
    assert torch.allclose(unmerged_out, merged_out, atol=1e-5)

    lora.unmerge()
    assert not lora.merged
    restored_out = lora(x)
    assert torch.allclose(unmerged_out, restored_out, atol=1e-5)
