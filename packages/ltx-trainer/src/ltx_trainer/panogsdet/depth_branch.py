"""Panoramic depth + semantic feature branch (Panoformer-style stub)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig


class PanoramicDepthBranch(nn.Module):
    """
    Lightweight ERP encoder producing depth map D and semantic features F.
    Weights may be frozen when mimicking pretrained Panoformer on Structured3D.
    """

    def __init__(self, cfg: PanoGSDetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoGSDetConfig()
        d = self.cfg.depth_feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Conv2d(d, 1, 3, padding=1)
        self.sem_head = nn.Conv2d(d, self.cfg.feature_dim, 1)
        if self.cfg.freeze_depth_branch:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.encoder(rgb)
        depth = torch.nn.functional.relu(self.depth_head(feat)) + 0.5
        sem = self.sem_head(feat)
        depth_up = torch.nn.functional.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        sem_up = torch.nn.functional.interpolate(sem, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return depth_up, sem_up
