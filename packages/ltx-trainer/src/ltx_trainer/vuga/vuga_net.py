"""VUGA full model (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.vuga.aff import AFFModule
from ltx_trainer.vuga.backbone import HierarchicalBackbone
from ltx_trainer.vuga.cae import CAEModule, QualityRegressor
from ltx_trainer.vuga.cmp import CMPStack
from ltx_trainer.vuga.config import VUGAConfig


class VUGA(nn.Module):
    """Viewport-unaware unified BOIQA / BIQA model."""

    def __init__(self, cfg: VUGAConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VUGAConfig()
        self.backbone = HierarchicalBackbone(self.cfg)
        self.cmp = CMPStack(self.cfg.stage_dims, self.cfg.cmp_dim)
        self.aff = AFFModule(self.cfg.cmp_dim)
        self.cae = CAEModule(self.cfg.stage_dims[-1])
        self.regressor = QualityRegressor(self.cfg.stage_dims[-1], self.cfg.cmp_dim)

    def preprocess(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.cfg.input_size:
            x = F.interpolate(
                x,
                size=(self.cfg.input_size, self.cfg.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.preprocess(x)
        feats = self.backbone(x)
        cmp_feats = self.cmp(feats)
        faff = self.aff(cmp_feats)
        fcae = self.cae(feats[-1])
        fcae_pool = F.adaptive_avg_pool2d(fcae, 1).flatten(1)
        faff_pool = F.adaptive_avg_pool2d(faff, 1).flatten(1)
        score = self.regressor(fcae_pool, faff_pool)
        return {"score": score, "faff": faff, "fcae": fcae}
