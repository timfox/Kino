"""EIC-LIE full network (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.eic_lie.eici import EICILayer
from ltx_trainer.eic_lie.iaef import IAEFLite
from ltx_trainer.eic_lie.retinex import illumination_prior


@dataclass
class EicLieConfig:
    base_channels: int = 32
    event_bins: int = 5
    eici_stages: tuple[int, ...] = (2, 2, 2)  # N1, N2, N3 block counts per paper
    iaef_lite: bool = True
    num_heads: int = 4


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EicLie(nn.Module):
    """Event-illumination collaborative low-light enhancement."""

    def __init__(self, cfg: EicLieConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EicLieConfig()
        c = self.cfg.base_channels
        bins = self.cfg.event_bins

        self.image_stem = _ConvBlock(3, c)
        self.event_stem = _ConvBlock(bins, c)
        self.illum_stem = _ConvBlock(1, c)

        self.eici_blocks = nn.ModuleList(
            [EICILayer(c, num_heads=self.cfg.num_heads) for _ in self.cfg.eici_stages]
        )
        self.iaef_blocks = nn.ModuleList([IAEFLite(c) for _ in self.cfg.eici_stages])

        self.refine = nn.Sequential(
            _ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )
        self.skip_proj = nn.Conv2d(3, 3, 1)

    def forward(self, low_image: Tensor, event_voxel: Tensor) -> Tensor:
        """
        Args:
            low_image: ``[B,3,H,W]`` low-light RGB in ``[0,1]``.
            event_voxel: ``[B,Bins,H,W]`` SBT representation.

        Returns:
            Enhanced RGB ``[B,3,H,W]`` in ``[0,1]``.
        """
        lp = illumination_prior(low_image)
        fi = self.image_stem(low_image)
        fe = self.event_stem(event_voxel)
        fl = self.illum_stem(lp)

        for eici, iaef in zip(self.eici_blocks, self.iaef_blocks, strict=True):
            fi, fe, fl = eici(fi, fe, fl)
            fe = iaef(fe, fl)

        fused = torch.cat([fi, fe], dim=1)
        residual = self.refine(fused)
        out = torch.sigmoid(self.skip_proj(low_image) + residual)
        return out.clamp(0, 1)
