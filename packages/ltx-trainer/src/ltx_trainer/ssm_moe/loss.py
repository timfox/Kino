"""Training losses for SSM-MoE models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ltx_trainer.ssm_moe.moe import load_balance_loss


def cross_entropy_lm(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal LM cross-entropy; ``labels`` same shape as ``input_ids``."""
    bsz, seqlen, vocab = logits.shape
    return F.cross_entropy(logits.reshape(bsz * seqlen, vocab), labels.reshape(bsz * seqlen))


def ssm_moe_total_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    router_logits: list[torch.Tensor],
    *,
    num_experts: int,
    load_balance_coeff: float = 0.01,
) -> dict[str, torch.Tensor]:
    """LM loss + summed MoE load-balancing aux losses."""
    lm = cross_entropy_lm(lm_logits, labels)
    aux = torch.tensor(0.0, device=lm_logits.device, dtype=lm_logits.dtype)
    for rl in router_logits:
        aux = aux + load_balance_loss(rl, num_experts)
    total = lm + load_balance_coeff * aux
    return {"total": total, "lm": lm, "load_balance": aux}
