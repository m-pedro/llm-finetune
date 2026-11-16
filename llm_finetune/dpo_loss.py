from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from llm_finetune.chat import IGNORE_INDEX


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Computes summed (or averaged) log-probabilities over non-masked target tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss_mask = labels != IGNORE_INDEX

    # Replace -100 with 0 for safe gather, then zero out masked positions
    safe_labels = torch.where(loss_mask, labels, torch.zeros_like(labels))
    per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    masked_logps = per_token_logps * loss_mask.to(per_token_logps.dtype)

    seq_logps = masked_logps.sum(dim=-1)
    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1).to(seq_logps.dtype)
        return seq_logps / token_counts
    return seq_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the Direct Preference Optimization (DPO) loss and implicit reward metrics."""
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    losses = (
        -F.logsigmoid(logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-logits) * label_smoothing
    )
    loss = losses.mean()

    with torch.no_grad():
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "loss": float(loss.item()),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
        "reward_margin": reward_margin,
        "reward_accuracy": reward_accuracy,
    }
    return loss, metrics
