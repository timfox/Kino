"""DPT-style progressive decoder + matting head (Sec. 3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _ResUnit(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32 if ch >= 32 else 1, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32 if ch >= 32 else 1, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return x + h


class DPTDecoder(nn.Module):
    """Three-stage 2× upsampling with backbone + upsampler skip connections."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.stage1 = _ResUnit(dim)
        self.stage2 = _ResUnit(dim)
        self.stage3 = _ResUnit(dim)
        self.merge_backbone = nn.Conv2d(dim * 2, dim, 1)
        self.merge_upsampler = nn.Conv2d(dim * 2, dim, 1)
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16 if dim >= 32 else 1, dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
        )

    def forward(
        self,
        fbam_map: Tensor,
        backbone_map: Tensor,
        upsampler_map: Tensor | None,
    ) -> Tensor:
        """Returns alpha logits at full resolution ``[B,1,H,W]``."""
        x = fbam_map
        # stage 1 → 1/8
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        bb = F.interpolate(backbone_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.merge_backbone(torch.cat([x, bb], dim=1))
        x = self.stage1(x)
        # stage 2 → 1/4
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        # stage 3 → 1/2
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if upsampler_map is not None:
            up = F.interpolate(upsampler_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge_upsampler(torch.cat([x, up], dim=1))
        x = self.stage3(x)
        # matting head → full res
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.head(x)
