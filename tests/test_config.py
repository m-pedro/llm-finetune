import math
import pytest
from llm_finetune.config import LoRAConfig


def test_lora_config_scaling_and_validation():
    cfg = LoRAConfig(r=16, lora_alpha=32.0)
    assert math.isclose(cfg.scaling, 2.0)

    rs_cfg = LoRAConfig(r=16, lora_alpha=32.0, use_rslora=True)
    assert math.isclose(rs_cfg.scaling, 32.0 / 4.0)

    with pytest.raises(ValueError, match="rank r must be > 0"):
        LoRAConfig(r=0)

    with pytest.raises(ValueError, match="power of 2"):
        LoRAConfig(nf4_block_size=50)
