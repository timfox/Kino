"""Training losses (Sec. 4.6, Eq. 18)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lumaflux.color import bt2020_luma, pq_eotf


@dataclass
class LumaFluxLossConfig:
    lambda_luma: float = 1.0
    lambda_rgb: float = 1.0
    lambda_spline: float = 0.1


class LumaFluxLoss:
    def __init__(self, cfg: LumaFluxLossConfig | None = None) -> None:
        self.cfg = cfg or LumaFluxLossConfig()

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        spline_logits: Tensor | None = None,
        spline_smooth_fn=None,
    ) -> tuple[Tensor, dict[str, float]]:
        pred_lin = pq_eotf(pred)
        tgt_lin = pq_eotf(target)
        y_pred = bt2020_luma(pred_lin / pred_lin.max().clamp(min=1e-3))
        y_tgt = bt2020_luma(tgt_lin / tgt_lin.max().clamp(min=1e-3))
        l_luma = F.l1_loss(y_pred, y_tgt)
        l_rgb = F.l1_loss(pred_lin, tgt_lin)
        l_spline = torch.zeros((), device=pred.device)
        if spline_logits is not None and spline_smooth_fn is not None:
            l_spline = spline_smooth_fn(spline_logits)
        total = (
            self.cfg.lambda_luma * l_luma
            + self.cfg.lambda_rgb * l_rgb
            + self.cfg.lambda_spline * l_spline
        )
        stats = {
            "loss_total": float(total.detach()),
            "loss_luma": float(l_luma.detach()),
            "loss_rgb": float(l_rgb.detach()),
            "loss_spline": float(l_spline.detach()),
        }
        return total, stats
