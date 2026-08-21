"""Two-stage TORM objectives (Sec. 3.3, Eq. 3–4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.torm.config import TORMConfig
from ltx_trainer.torm.latent import latent_alignment_loss


def answer_token_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """L_ans: standard next-token CE on answer span."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def stage1_loss(
    z: Tensor,
    g: Tensor,
    ans_logits: Tensor,
    ans_labels: Tensor,
    *,
    cfg: TORMConfig,
) -> tuple[Tensor, dict[str, float]]:
    """L_stage1 = L_ans + λ L_latent."""
    l_latent = latent_alignment_loss(z, g)
    l_ans = answer_token_loss(ans_logits, ans_labels)
    total = l_ans + cfg.latent_align_weight * l_latent
    return total, {
        "l_ans": float(l_ans.item()),
        "l_latent": float(l_latent.item()),
        "l_total": float(total.item()),
    }


def stage2_loss(ans_logits: Tensor, ans_labels: Tensor) -> Tensor:
    """Stage II: answer-only (no latent supervision)."""
    return answer_token_loss(ans_logits, ans_labels)
