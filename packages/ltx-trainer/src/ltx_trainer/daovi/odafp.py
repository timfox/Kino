"""Omnidirectional Depth-Assisted Feature Propagation (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.daovi.distortion import DistortionGuidanceGenerator, distortion_map


class ACDConvStub(nn.Module):
    """Adaptively combined dilated convolution stub [42]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=d, dilation=d) for d in (1, 2, 3)]
        )
        self.weight = nn.Conv2d(channels * 3, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [b(x) for b in self.branches]
        cat = torch.cat(outs, dim=1)
        w = F.softmax(self.weight(cat), dim=1)
        return sum(w[:, i : i + 1] * outs[i] for i in range(3))


class ODAFPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dgg = DistortionGuidanceGenerator(hidden=dim // 2)
        self.depth_proj = nn.Conv2d(dim // 2, 2, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2 + 4, dim, 3, padding=1),
            ACDConvStub(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.offset_head = nn.Conv2d(dim, 2, 3, padding=1)
        self.mask_head = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(
        self,
        feat: Tensor,
        feat_adj: Tensor,
        flow: Tensor,
        depth_feat: Tensor,
        mask: Tensor,
        unfilled: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        dmap = distortion_map(height, width, device=feat.device)
        guidance = self.dgg(dmap)
        warped_adj = feat_adj  # stub: skip full DCN warp
        depth2 = self.depth_proj(depth_feat)
        x = torch.cat([feat, warped_adj, flow[:, :2], depth2], dim=1)
        h = self.fuse(x)
        g = guidance.mean(dim=(2, 3), keepdim=True)
        mod = torch.sigmoid(self.mask_head(h) * g)
        return feat + mod * h
