"""Training losses for EIC-LIE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EicLieLossConfig:
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    perceptual_weight: float = 0.0


class EicLieLoss(nn.Module):
    """L1 + (1 - SSIM) on enhanced vs normal-light GT."""

    def __init__(self, cfg: EicLieLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EicLieLossConfig()

    def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        l1 = F.l1_loss(pred, target)
        ssim_val = _ssim_torch(pred, target)
        loss = self.cfg.l1_weight * l1 + self.cfg.ssim_weight * (1.0 - ssim_val)
        return loss, {
            "loss_total": float(loss.detach()),
            "l1": float(l1.detach()),
            "ssim": float(ssim_val.detach()),
        }


def _ssim_torch(x: Tensor, y: Tensor, window_size: int = 11) -> Tensor:
    c = x.shape[1]
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x**2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()
