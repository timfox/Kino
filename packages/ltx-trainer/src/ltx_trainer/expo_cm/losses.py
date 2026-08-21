"""Consistency training loss (Eq. 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.expo_cm.trajectory import EACTConfig, sample_eact_state


@dataclass
class ConsistencyLossConfig:
    t_min: float = 0.05
    t_max: float = 1.0
    eact: EACTConfig | None = None


class ConsistencyLoss:
    def __init__(self, cfg: ConsistencyLossConfig | None = None) -> None:
        self.cfg = cfg or ConsistencyLossConfig()

    def __call__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        x0: Tensor,
        y0: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.cfg
        eact = cfg.eact or EACTConfig(t_max=cfg.t_max)
        b = x0.shape[0] if x0.dim() == 4 else 1
        device = x0.device
        t = torch.rand(b, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
        t2 = t * torch.rand(b, device=device)
        eps = torch.randn_like(x0 if x0.dim() == 4 else x0.unsqueeze(0))
        eps2 = torch.randn_like(eps)
        x0b = x0.unsqueeze(0) if x0.dim() == 3 else x0
        y0b = y0.unsqueeze(0) if y0.dim() == 3 else y0
        xt, _ = sample_eact_state(x0b, y0b, t, eps, cfg=eact)
        xt2, _ = sample_eact_state(x0b, y0b, t2, eps2, cfg=eact)
        pred = model(xt, t, y0b)
        with torch.no_grad():
            tgt = target_model(xt2, t2, y0b)
        loss = F.mse_loss(pred, tgt)
        return loss, {"loss_ct": float(loss.detach())}
