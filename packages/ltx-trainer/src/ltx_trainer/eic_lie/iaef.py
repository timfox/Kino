"""Illumination-aware Event Filter (IAEF, Sec. 3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IAEFLite(nn.Module):
    """
    Lightweight IAEF: separable illumination kernels + event-driven deformable mixing.

    Full paper uses n×n kernels with offsets; this variant uses 3×3 separable filters
    for stable training on synthetic/RLE drops without CUDA custom ops.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        k = kernel_size
        self.k = k
        pad = k // 2
        self.kh = nn.Conv2d(channels, channels, (1, k), padding=(0, pad), groups=channels, bias=False)
        self.kv = nn.Conv2d(channels, channels, (k, 1), padding=(pad, 0), groups=channels, bias=False)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, event_feat: Tensor, illum_feat: Tensor) -> Tensor:
        kh = self.kh(illum_feat)
        kv = self.kv(illum_feat)
        kernel = kh * kv
        w = self.weight_net(event_feat)
        offset = self.offset_net(event_feat) * 0.1
        b, c, h, wid = event_feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=event_feat.device, dtype=event_feat.dtype),
            torch.linspace(-1, 1, wid, device=event_feat.device, dtype=event_feat.dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        grid = grid + offset.permute(0, 2, 3, 1)
        sampled = F.grid_sample(event_feat, grid, align_corners=True, mode="bilinear")
        filtered = kernel * w * sampled
        return event_feat + self.out(filtered)
