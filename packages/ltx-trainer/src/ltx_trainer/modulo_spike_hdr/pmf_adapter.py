"""Pyramidal multi-frequency adapter stub (PMF-Adapter, Eq. 10)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.modulo_spike_hdr.lar import lar_gradient_features, lar_laplacian_poisson


class _CBAMLite(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(ch, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.ca(x) * self.sa(x)


class PMFAdapter(nn.Module):
    """Fuses Im, R(∇Im), P(R(ΔIm)) into multiscale features Fa_1..4."""

    def __init__(self, in_ch: int = 3, base: int = 32, period: float = 256.0) -> None:
        super().__init__()
        self.period = period
        # Im (C) + R(∇Im) (2C) + P(R(ΔIm)) (C)
        fuse_in = in_ch * 4
        self.stem = nn.Sequential(
            nn.Conv2d(fuse_in, base, 3, padding=1),
            nn.ReLU(inplace=True),
            _CBAMLite(base),
        )
        self.scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base, base, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base, base, 2, stride=2),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.heads = nn.ModuleList([nn.Conv2d(base, base, 1) for _ in range(4)])

    def _frequency_stack(self, modulo: Tensor) -> Tensor:
        squeeze = modulo.dim() == 3
        if squeeze:
            modulo = modulo.unsqueeze(0)
        period = self.period
        grad = lar_gradient_features(modulo, period)
        pois = lar_laplacian_poisson(modulo, period)
        if grad.dim() == 3:
            grad = grad.unsqueeze(0)
        if pois.shape[1] != modulo.shape[1]:
            pois = pois[:, : modulo.shape[1]]
        x = torch.cat([modulo, grad, pois], dim=1)
        return x.squeeze(0) if squeeze else x

    def forward(self, modulo: Tensor) -> list[Tensor]:
        x = self._frequency_stack(modulo)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.stem(x)
        feats = [self.heads[0](h)]
        cur = h
        for i, block in enumerate(self.scales):
            cur = block(cur)
            feats.append(self.heads[i + 1](cur))
        return feats
