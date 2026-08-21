"""Training losses: tone-mapped ℓ1, warp, shift variance."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LuckyHdrLoss(nn.Module):
    def __init__(self, *, lambda_warp: float = 0.5, lambda_shift: float = 0.1) -> None:
        super().__init__()
        self.lambda_warp = lambda_warp
        self.lambda_shift = lambda_shift

    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        *,
        warp_terms: list[Tensor] | None = None,
        shifts: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        loss_pred = (pred - gt).abs().mean()
        loss_warp = torch.stack(warp_terms).mean() if warp_terms else pred.new_tensor(0.0)
        if shifts:
            stacked = torch.stack([s.reshape(-1) for s in shifts], dim=0)
            loss_shift = stacked.var(dim=0).mean()
        else:
            loss_shift = pred.new_tensor(0.0)
        total = loss_pred + self.lambda_warp * loss_warp + self.lambda_shift * loss_shift
        stats = {
            "loss_pred": float(loss_pred.detach()),
            "loss_warp": float(loss_warp.detach()),
            "loss_shift": float(loss_shift.detach()),
            "loss_total": float(total.detach()),
        }
        return total, stats
