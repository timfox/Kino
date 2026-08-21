"""Depth-based 3D reconstruction evaluation stub (Sec. 3.5 iii)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.ob3d.config import OB3DConfig
from ltx_trainer.ob3d.metrics import depth_abs_rel, depth_delta125, depth_rmse, sky_mask
from ltx_trainer.ob3d.synthetic import synthetic_depth, synthetic_rgb


class DepthReconStub(nn.Module):
    """Predict depth from ERP RGB; stand-in for COLMAP / NeuS* depth render."""

    def __init__(self, cfg: OB3DConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OB3DConfig()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        return F.softplus(self.net(rgb)) + 0.1


def evaluate_depth_reconstruction(cfg: OB3DConfig | None = None) -> dict[str, float]:
    cfg = cfg or OB3DConfig()
    model = DepthReconStub(cfg)
    rgb = synthetic_rgb(cfg)
    gt = synthetic_depth(cfg)
    with torch.no_grad():
        pred = model(rgb)
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    mask = sky_mask(gt.squeeze(), cfg.sky_depth_threshold)
    return {
        "RMSE": float(depth_rmse(pred, gt, mask).item()),
        "AbsRel": float(depth_abs_rel(pred, gt, mask).item()),
        "delta125": float(depth_delta125(pred, gt, mask).item()),
    }
