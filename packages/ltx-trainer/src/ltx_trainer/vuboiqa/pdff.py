"""Progressive deformation-unaware feature fusion stub (Sec. 3.3, Eq. 3–9)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _match_spatial(a: Tensor, b: Tensor) -> Tensor:
    if a.shape[-2:] == b.shape[-2:]:
        return b
    return F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)


class DeformConvStub(nn.Module):
    """Offset-style conv stub (DCN/DCNv3 placeholder)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.offset = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x + 0.1 * self.offset(x))


class DAAStub(nn.Module):
    """Distortion-aware attention (Eq. 6–8)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(ch, ch, 1)
        self.dcn = DeformConvStub(ch)
        self.spatial = nn.Conv2d(ch, 1, 7, padding=3, stride=3)

    def forward(self, x: Tensor) -> Tensor:
        n = self.dcn(self.reduce(x))
        ca = n * torch.sigmoid(n.mean(dim=(2, 3), keepdim=True))
        sa = torch.sigmoid(self.spatial(ca))
        sa = _match_spatial(ca, sa)
        return F.max_pool2d(ca * sa, kernel_size=2, stride=2)


class PDFFModule(nn.Module):
    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.stages = nn.ModuleList([DeformConvStub(ch) for _ in range(3)])
        self.fuse01 = nn.Conv2d(ch, ch, 1, stride=2)
        self.daa = nn.ModuleList([DAAStub(ch) for _ in range(2)])

    def forward(self, feats: list[Tensor]) -> Tensor:
        adjusted = [self.stages[i](feats[i]) for i in range(min(3, len(feats)))]
        while len(adjusted) < 3:
            adjusted.append(adjusted[-1])
        f01 = self.fuse01(adjusted[0]) + _match_spatial(self.fuse01(adjusted[0]), adjusted[1])
        f12 = adjusted[1] + _match_spatial(adjusted[1], adjusted[2])
        f01 = self.stages[1](f01)
        f12 = self.stages[2](f12)
        o0 = self.daa[0](f01)
        o1 = self.daa[1](f12)
        return o0 + _match_spatial(o0, o1)
