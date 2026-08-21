"""FogNet losses: InfoNCE + temporal alignment (Sec. 4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FogNetLossConfig:
    lambda_clean: float = 0.4
    beta_temp: float = 0.1
    temperature: float = 0.07


def _clip_contrastive(video: Tensor, text: Tensor, *, temperature: float) -> tuple[Tensor, Tensor]:
    """InfoNCE video↔text (Eq. 8–9). ``video``, ``text``: ``(B,D)``."""
    v = F.normalize(video, dim=-1)
    t = F.normalize(text, dim=-1)
    logits = (v @ t.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    lt2v = F.cross_entropy(logits.T, labels)
    lv2t = F.cross_entropy(logits, labels)
    return lt2v, lv2t


class FogNetLoss:
    def __init__(self, cfg: FogNetLossConfig | None = None) -> None:
        self.cfg = cfg or FogNetLossConfig()

    def __call__(
        self,
        *,
        fog_emb: Tensor,
        clean_emb: Tensor,
        text_emb: Tensor,
        temp_loss: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        lt2v_f, lv2t_f = _clip_contrastive(fog_emb, text_emb, temperature=self.cfg.temperature)
        lf = lt2v_f + lv2t_f
        lt2v_c, lv2t_c = _clip_contrastive(clean_emb, text_emb, temperature=self.cfg.temperature)
        lc = lt2v_c + lv2t_c
        total = lf + self.cfg.lambda_clean * lc + self.cfg.beta_temp * temp_loss
        return total, {
            "loss_total": float(total.detach()),
            "loss_fog": float(lf.detach()),
            "loss_clean": float(lc.detach()),
            "loss_temp": float(temp_loss.detach()),
        }
