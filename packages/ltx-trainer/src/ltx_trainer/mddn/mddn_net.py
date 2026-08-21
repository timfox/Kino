"""MDDN network stub (Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.mddn.config import MddnConfig
from ltx_trainer.mddn.distortion import erp_distortion_map
from ltx_trainer.mddn.mdde import MDDE
from ltx_trainer.mddn.mff import MultiLevelFeatureFusion, fuse_by_addition


class MDDNStub(nn.Module):
    """Shallow → MDDE+MFF → reconstruct (×scale via interpolate)."""

    def __init__(self, cfg: MddnConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MddnConfig()
        c = self.cfg.hidden_dim
        ic = self.cfg.in_channels
        self.shallow = nn.Conv2d(ic, c, 3, padding=1)
        self.mdde = MDDE(self.cfg)
        n_br = len(self.cfg.branches)
        self.mff = MultiLevelFeatureFusion(c, num_branches=max(n_br, 1))
        self.tail = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, ic, 3, padding=1),
        )

    def forward(self, lr: Tensor) -> dict[str, Tensor]:
        b, _, h, w = lr.shape
        d_map = erp_distortion_map(h, w, device=lr.device)
        f = self.shallow(lr)
        branch_out = self.mdde(f, d_map)
        feats = list(branch_out.values())
        if self.cfg.use_mff:
            fused = self.mff(feats)
        else:
            fused = fuse_by_addition(feats)
        residual = self.tail(fused)
        sr = lr + residual
        if self.cfg.scale > 1:
            sr = nn.functional.interpolate(
                sr,
                scale_factor=self.cfg.scale,
                mode="bilinear",
                align_corners=False,
            )
        return {
            "sr": sr.clamp(0.0, 1.0),
            "distortion": d_map,
            "fused": fused,
        }
