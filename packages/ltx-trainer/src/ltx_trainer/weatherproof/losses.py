"""Supervised + unsupervised losses (Sec. 2.2, 2.5–2.6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.weatherproof.classes import IGNORE_LABEL


@dataclass
class SemiSupervisedLossConfig:
    lambda_unsup: float = 1.0
    confidence_tau: float = 0.95


class SemiSupervisedLoss:
    def __init__(self, cfg: SemiSupervisedLossConfig | None = None) -> None:
        self.cfg = cfg or SemiSupervisedLossConfig()

    def supervised(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target.long(), ignore_index=IGNORE_LABEL)

    def unsupervised(
        self,
        logits1: Tensor,
        logits2: Tensor,
        pseudo1: Tensor,
        pseudo2: Tensor,
        conf1: Tensor,
        conf2: Tensor,
    ) -> Tensor:
        ce1 = F.cross_entropy(logits1, pseudo1.long(), reduction="none")
        ce2 = F.cross_entropy(logits2, pseudo2.long(), reduction="none")
        m1 = conf1.float()
        m2 = conf2.float()
        denom = (m1.sum() + m2.sum()).clamp(min=1.0)
        return (ce1 * m1).sum() / denom + (ce2 * m2).sum() / denom

    def __call__(
        self,
        clean_logits: Tensor,
        clean_target: Tensor,
        *,
        strong_logits1: Tensor | None = None,
        strong_logits2: Tensor | None = None,
        pseudo1: Tensor | None = None,
        pseudo2: Tensor | None = None,
        conf1: Tensor | None = None,
        conf2: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        lc = self.supervised(clean_logits, clean_target)
        ld = torch.tensor(0.0, device=clean_logits.device)
        if strong_logits1 is not None and pseudo1 is not None:
            ld = self.unsupervised(strong_logits1, strong_logits2, pseudo1, pseudo2, conf1, conf2)
        loss = lc + self.cfg.lambda_unsup * ld
        return loss, {
            "loss_total": float(loss.detach()),
            "loss_clean": float(lc.detach()),
            "loss_degraded": float(ld.detach()),
        }
