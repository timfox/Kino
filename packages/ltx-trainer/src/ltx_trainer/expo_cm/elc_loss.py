"""Exposure-guided Luminance-Chromaticity loss (Eq. 12–15)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.expo_cm.exposure_mask import compute_exposure_masks, luminance
from ltx_trainer.expo_cm.lab_color import rgb_to_lab


def charbonnier(x: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


@dataclass
class ELCLossConfig:
    lambda_l0: float = 1.0
    lambda_c0: float = 1.0
    kappa_lo_l: float = 3.0
    kappa_hi_l: float = 1.0
    kappa_hi_c: float = 3.0
    kappa_lo_c: float = 0.5
    alpha: float = 1.0
    tau_s: float = 0.2
    tau_h: float = 0.8
    delta_s: float = 0.1
    delta_h: float = 0.1
    c0: float = 0.05


class ELCLoss:
    def __init__(self, cfg: ELCLossConfig | None = None) -> None:
        self.cfg = cfg or ELCLossConfig()

    def _sigmoid_gate(self, x: Tensor, tau: float, delta: float) -> Tensor:
        return torch.sigmoid((tau - x) / delta)

    def weights(
        self,
        ldr: Tensor,
        gt_lab: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        masks = compute_exposure_masks(ldr)
        wunder = masks["wunder"]
        wover = masks["wover"]
        y = masks["luminance"]
        if wunder.dim() == 3:
            wunder = wunder.unsqueeze(0)
            wover = wover.unsqueeze(0)
            y = y.unsqueeze(0)
        wunder_a = wunder.pow(cfg.alpha)
        wover_a = wover.pow(cfg.alpha)
        s_y = self._sigmoid_gate(y, cfg.tau_s, cfg.delta_s)
        h_y = self._sigmoid_gate(y - cfg.tau_h, 0.0, cfg.delta_h)
        _, a0, b0 = gt_lab
        if a0.dim() == 3:
            a0 = a0.unsqueeze(0)
            b0 = b0.unsqueeze(0)
        c0 = torch.sqrt(a0 * a0 + b0 * b0 + 1e-8)
        a_spec = 1.0 / (1.0 + c0 / cfg.c0)
        w_l = cfg.lambda_l0 * (1.0 + cfg.kappa_lo_l * s_y * wunder_a + cfg.kappa_hi_l * a_spec * wover_a)
        w_c = cfg.lambda_c0 * (
            cfg.kappa_hi_c * wover_a * (1.0 - a_spec) * h_y + cfg.kappa_lo_c * wunder_a * (1.0 - s_y)
        )
        return w_l, w_c

    def __call__(
        self,
        pred: Tensor,
        gt: Tensor,
        ldr: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            ldr = ldr.unsqueeze(0)
        pl, pa, pb = rgb_to_lab(pred.clamp(0.0, 1.0))
        gl, ga, gb = rgb_to_lab(gt.clamp(0.0, 1.0))
        d_l = pl - gl
        d_c = torch.sqrt((pa - ga) ** 2 + (pb - gb) ** 2 + 1e-8)
        w_l, w_c = self.weights(ldr, (gl, ga, gb))
        loss_l = (w_l * charbonnier(d_l)).mean()
        loss_c = (w_c * charbonnier(d_c)).mean()
        loss = loss_l + loss_c
        return loss, {"loss_elc": float(loss.detach()), "loss_l": float(loss_l.detach()), "loss_c": float(loss_c.detach())}
