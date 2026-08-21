"""FastViDAR depth head stub (Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.fastvidar.aha import AHAStack
from ltx_trainer.fastvidar.config import FastViDARConfig
from ltx_trainer.fastvidar.erp_fusion import erp_mean_fusion


class FastViDARStub(nn.Module):
    def __init__(self, cfg: FastViDARConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FastViDARConfig()
        c = self.cfg.hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.aha = AHAStack(self.cfg)
        self.depth_head = nn.Conv2d(c, 1, 3, padding=1)
        self.conf_head = nn.Conv2d(c, 1, 3, padding=1)

    def encode_frames(self, frames: Tensor) -> Tensor:
        """frames [B, S, 3, H, W] → features [B, S, C, h, w]."""
        b, s, _, h, w = frames.shape
        x = frames.view(b * s, 3, h, w)
        x = self.stem(x)
        _, c, hh, ww = x.shape
        return x.view(b, s, c, hh, ww)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        feat = self.encode_frames(frames)
        feat = self.aha(feat)
        b, s, c, hh, ww = feat.shape
        depths = []
        confs = []
        for i in range(s):
            d = self.depth_head(feat[:, i]).squeeze(1).relu() + 0.1
            cf = torch.sigmoid(self.conf_head(feat[:, i]).squeeze(1))
            depths.append(d)
            confs.append(cf)
        per_cam = torch.stack(depths, dim=1)
        per_conf = torch.stack(confs, dim=1)

        fused_list = []
        for bi in range(b):
            mask = torch.ones(s, hh, ww, device=frames.device)
            fused, _ = erp_mean_fusion(per_cam[bi], mask)
            fused_list.append(fused)
        fusion = torch.stack(fused_list, dim=0)

        return {
            "per_camera_depth": per_cam,
            "confidence": per_conf,
            "fusion_depth": fusion,
        }
