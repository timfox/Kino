"""Multi-level Distortion-aware Deformable Extractor (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mddn.config import MddnConfig
from ltx_trainer.mddn.distortion import erp_distortion_map


class OffsetNet(nn.Module):
    """Distortion-conditioned offset predictor (stub)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, d_map: Tensor) -> Tensor:
        return self.net(d_map)


class DilatedDeformConvStub(nn.Module):
    """D4C: dilated conv + distortion offsets (level-2/3)."""

    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.offset = OffsetNet()

    def forward(self, feat: Tensor, d_map: Tensor) -> Tensor:
        off = self.offset(d_map)
        shifted = feat + 0.01 * off[:, :1] * d_map
        return self.conv(shifted)


class DDCAStub(nn.Module):
    """Distortion-aware deformable cross-attention (level-1, d=1)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.offset = OffsetNet()
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, feat: Tensor, d_map: Tensor) -> Tensor:
        off = self.offset(d_map)
        warped = feat + 0.01 * off[:, :1] * d_map
        q = self.q(feat)
        k = self.k(warped)
        v = self.v(warped)
        attn = torch.softmax(
            (q.flatten(2) @ k.flatten(2).transpose(-1, -2)) / (feat.shape[1] ** 0.5),
            dim=-1,
        )
        out = (attn @ v.flatten(2)).view_as(feat)
        return self.proj(out)


class MDDE(nn.Module):
    """Parallel DDCA (d=1) + D4C (d=2) + D4C (d=3)."""

    def __init__(self, cfg: MddnConfig) -> None:
        super().__init__()
        c = cfg.hidden_dim
        self.branches = nn.ModuleDict()
        if 1 in cfg.branches:
            self.branches["1"] = DDCAStub(c)
        if 2 in cfg.branches:
            self.branches["2"] = DilatedDeformConvStub(c, dilation=2)
        if 3 in cfg.branches:
            self.branches["3"] = DilatedDeformConvStub(c, dilation=3)

    def forward(self, feat: Tensor, d_map: Tensor) -> dict[str, Tensor]:
        outs: dict[str, Tensor] = {}
        if "1" in self.branches:
            outs["f1"] = self.branches["1"](feat, d_map)
        if "2" in self.branches:
            outs["f2"] = self.branches["2"](feat, d_map)
        if "3" in self.branches:
            outs["f3"] = self.branches["3"](feat, d_map)
        return outs
