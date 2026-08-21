"""ERP-based feature encoder (Sec. III-B): conv finest + ResNet34-style pyramid stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cross360.config import NUM_SCALES, Cross360Config


class ERPEncoderStub(nn.Module):
    """Multi-scale F^ERP_s for s=1..S (coarse→fine indexing in pipeline)."""

    def __init__(self, cfg: Cross360Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Cross360Config()
        c = self.cfg.embed_dim
        self.fine_conv = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
            ]
        )
        self.out_channels = [c, c * 2, c * 2, c * 4, c * 4]

    def forward(self, erp: Tensor) -> list[Tensor]:
        feats: list[Tensor] = []
        x = self.fine_conv(erp)
        feats.append(x)
        for block in self.down:
            x = block(x)
            feats.append(x)
        while len(feats) < NUM_SCALES:
            feats.append(feats[-1])
        return feats[:NUM_SCALES]
