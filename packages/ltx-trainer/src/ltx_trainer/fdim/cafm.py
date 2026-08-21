"""Content-Adaptive Feature-Distance Modeling (CAFM, Eq. 5–7)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OffsetGenerator(nn.Module):
    """Predict 2D sampling offsets from reference features (deformable conv proxy)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 3, padding=1),
        )

    def forward(self, ref: Tensor) -> Tensor:
        return self.net(ref).tanh() * 0.1


class CAFM(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.offset_gen = OffsetGenerator(ch)
        self.conv = nn.Conv2d(ch * 3, ch, 3, padding=1)

    def _deform_sample(self, x: Tensor, ref: Tensor) -> Tensor:
        b, c, h, w = x.shape
        off = self.offset_gen(ref)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        off_n = off.permute(0, 2, 3, 1)
        off_n = torch.stack((off_n[..., 0] / max(w - 1, 1) * 2, off_n[..., 1] / max(h - 1, 1) * 2), dim=-1)
        grid = (base + off_n).clamp(-1.0, 1.0)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, ref: Tensor, dist: Tensor) -> Tensor:
        err = (ref - dist) ** 2
        cat = torch.cat([ref, dist, err], dim=1)
        aligned = self._deform_sample(cat, ref)
        return F.relu(self.conv(aligned))
