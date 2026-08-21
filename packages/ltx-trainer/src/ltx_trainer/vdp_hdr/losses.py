"""PU-21 fusion loss (Talegaonkar et al. eq. 9–10; X2HDR PU-21 via hdr_ingest)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.hdr_ingest import pu21_forward_linear_abs


class FusionLoss(nn.Module):
    """L2 in PU-21 on peak-normalized linear HDR (scale-invariant up to global gain)."""

    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def _as_bchw(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            return x.unsqueeze(0)
        return x

    def _normalize_peak(self, linear: Tensor) -> Tensor:
        x = self._as_bchw(linear)
        peak = x.amax(dim=(1, 2, 3), keepdim=True).clamp(min=self.eps)
        return (x / peak).squeeze(0) if linear.ndim == 3 else x / peak

    def _pu21_rgb(self, linear: Tensor) -> Tensor:
        """Apply PU-21 per channel after mapping peak to X2HDR-style 4000 cd/m² scale."""
        x = self._as_bchw(linear)
        peak = x.amax(dim=(1, 2, 3), keepdim=True).clamp(min=self.eps)
        l_abs = (x / peak) * 4000.0
        out = pu21_forward_linear_abs(l_abs)
        return out.squeeze(0) if linear.ndim == 3 else out

    def forward(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, dict[str, float]]:
        p_in = self._as_bchw(pred.clamp(min=0.0))
        g_in = self._as_bchw(gt.clamp(min=0.0))
        p = self._pu21_rgb(p_in)
        g = self._pu21_rgb(g_in)
        loss = torch.nn.functional.mse_loss(p, g)
        stats = {
            "fusion_mse": float(loss.detach()),
            "pred_peak": float(p_in.max().detach()),
        }
        return loss, stats
