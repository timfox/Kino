"""MSFET-E2V U-Net-style encoder–decoder (Fig. 2) — smoke-scale implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.msfet_e2v.blocks import CDAMBlock, DownConv, ResidualGuidedDecoder, WaveletSkipBlock
from ltx_trainer.msfet_e2v.config import MSFETE2VConfig


class MSFETE2V(nn.Module):
    """Multiscale frequency-enhanced transformer for event-to-video reconstruction."""

    def __init__(self, cfg: MSFETE2VConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MSFETE2VConfig()
        self.cfg = cfg
        c = cfg.base_channels
        d = cfg.encoder_depth
        self.head = nn.Conv2d(cfg.voxel_bins, c, 3, padding=1)
        ch = c
        self.downs = nn.ModuleList()
        self.cdams = nn.ModuleList()
        self.wsbs = nn.ModuleList()
        self.enc_channels: list[int] = []
        for i in range(d):
            out_ch = c * (2 ** (i + 1))
            self.downs.append(DownConv(ch, out_ch))
            self.cdams.append(CDAMBlock(out_ch, cfg.embed_dim))
            self.wsbs.append(WaveletSkipBlock(out_ch))
            self.enc_channels.append(out_ch)
            ch = out_ch
        self.rgds = nn.ModuleList()
        for i in range(d - 1, -1, -1):
            in_ch = self.enc_channels[i]
            out_ch = self.enc_channels[i - 1] if i > 0 else c
            self.rgds.append(ResidualGuidedDecoder(in_ch, out_ch))
        self.pred = nn.Conv2d(c, 1, 1)

    def forward(self, voxel: Tensor) -> Tensor:
        """voxel: (B, B_bins, H, W) -> intensity (B, 1, H, W)."""
        h = self.head(voxel)
        skips: list[Tensor] = []
        for down, cdam, wsb in zip(self.downs, self.cdams, self.wsbs, strict=True):
            h = cdam(down(h))
            skips.append(wsb(h))
        z = h
        for i, rgd in enumerate(self.rgds):
            z = rgd(z)
            skip_idx = len(skips) - 2 - i
            if skip_idx >= 0:
                z = z + skips[skip_idx]
        return self.pred(z)
