"""Frozen SwinV2-T hierarchical feature stub (Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.vuga.config import VUGAConfig


class HierarchicalBackbone(nn.Module):
    """Four-stage feature pyramid mimicking SwinV2-T stages."""

    def __init__(self, cfg: VUGAConfig) -> None:
        super().__init__()
        dims = cfg.stage_dims
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, dims[0], 4, stride=4, padding=0),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        in_d = dims[0]
        for i, out_d in enumerate(dims):
            stride = 2 if i > 0 else 1
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(in_d, out_d, 3, stride=stride, padding=1),
                    nn.GELU(),
                    nn.Conv2d(out_d, out_d, 3, padding=1),
                    nn.GELU(),
                )
            )
            in_d = out_d
        if cfg.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        feats: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
