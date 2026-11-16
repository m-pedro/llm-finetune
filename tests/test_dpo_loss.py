import math
import torch
from llm_finetune.chat import IGNORE_INDEX
from llm_finetune.dpo_loss import dpo_loss, sequence_log_probs


def test_dpo_loss_at_initialization_and_positive_margin_gradient():
    # When policy == reference, log-ratios are 0, so DPO loss must equal -log(sigmoid(0)) = ln(2)
    zero = torch.zeros(4)
    loss_init, metrics_init = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert math.isclose(loss_init.item(), math.log(2.0), rel_tol=1e-5)
    assert math.isclose(metrics_init["reward_margin"], 0.0, abs_tol=1e-6)

    # When policy prefers chosen over rejected, loss < ln(2) and accuracy == 1.0
    pol_c = torch.tensor([2.0, 1.5], requires_grad=True)
    pol_r = torch.tensor([-1.0, -0.5], requires_grad=True)
    ref_c = torch.tensor([0.0, 0.0])
    ref_r = torch.tensor([0.0, 0.0])

    loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.5)
    assert loss.item() < math.log(2.0)
    assert metrics["reward_margin"] > 0.0
    assert metrics["reward_accuracy"] == 1.0

    loss.backward()
    # Gradient wrt chosen log-probs must be negative (pushing chosen log-prob UP)
    assert torch.all(pol_c.grad < 0.0)
    # Gradient wrt rejected log-probs must be positive (pushing rejected log-prob DOWN)
    assert torch.all(pol_r.grad > 0.0)


def test_sequence_log_probs_ignores_masked_prompt_tokens():
    logits = torch.randn(2, 5, 16)
    labels1 = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5], [IGNORE_INDEX, 1, 2, 3, 4]])
    logits_perturbed = logits.clone()
    # Perturb logits at position 0 (which is masked in both sequences)
    logits_perturbed[:, 0, :] += 100.0

    lp1 = sequence_log_probs(logits, labels1)
    lp2 = sequence_log_probs(logits_perturbed, labels1)
    assert torch.allclose(lp1, lp2, atol=1e-6)
