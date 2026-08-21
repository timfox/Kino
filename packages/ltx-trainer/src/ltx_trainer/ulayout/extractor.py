"""Dual-branch ResNet-50 + 1D conv feature extractor stub (Sec. 3.3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.ulayout.config import ULayoutConfig


class Branch1DConv(nn.Module):
    """Three 1D convs (4×1, 2×1, 2×1) per HorizonNet/LGT-Net design."""

    def __init__(self, in_ch: int, out_w: int, circular: bool) -> None:
        super().__init__()
        self.circular = circular
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.out_w = out_w

    def forward(self, x: Tensor) -> Tensor:
        if self.circular:
            x = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        y = self.conv(x)
        if self.circular:
            y = y[..., 1:-1]
        y = self.proj(y)
        return torch.nn.functional.interpolate(y, size=(y.shape[-2], self.out_w), mode="bilinear", align_corners=False)


class DualBranchExtractor(nn.Module):
    """Shared backbone stub + separate 1D-conv branches for pano vs perspective."""

    def __init__(self, cfg: ULayoutConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = 64
        self.shared = nn.Sequential(
            nn.Conv2d(3, ch, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pano_branch = Branch1DConv(ch, cfg.pano_feature_w, circular=True)
        self.pp_branch = Branch1DConv(ch, cfg.pp_feature_w, circular=False)

    def forward(self, pano: Tensor, pp: Tensor) -> tuple[Tensor, Tensor]:
        fp = self.shared(pano)
        fpp = self.shared(pp)
        return self.pano_branch(fp), self.pp_branch(fpp)
