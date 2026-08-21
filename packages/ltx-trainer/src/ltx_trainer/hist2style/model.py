"""Hist2Style dual-branch ConvNeXt + cross-attention → bilateral grid (Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hist2style.bilateral import ChannelLUT, slice_bilateral_grid
from ltx_trainer.hist2style.config import Hist2StyleConfig
from ltx_trainer.hist2style.histogram import marginal_histogram


class ConvNeXtBlock2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pw = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = self.dw(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pw(y)
        y = y.permute(0, 3, 1, 2)
        return x + y


class Hist2Style(nn.Module):
    def __init__(self, cfg: Hist2StyleConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Hist2StyleConfig()
        c = self.cfg
        self.content_stem = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.content_blocks = nn.Sequential(ConvNeXtBlock2D(32), ConvNeXtBlock2D(32))
        self.hist_encoder = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
        )
        self.cross_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.head = nn.Conv2d(64, c.affine_dim + 1, 1)
        self.lut = ChannelLUT()
        self.guidance_proj = nn.Conv2d(3, 1, 1)

    def forward(self, content: Tensor, style_hist: Tensor | None = None) -> Tensor:
        if style_hist is None:
            style_hist = marginal_histogram(content, self.cfg.hist_bins)
        feat = self.content_blocks(self.content_stem(content))
        b, _, h, w = feat.shape
        hist_feat = self.hist_encoder(style_hist).mean(dim=-1, keepdim=True).expand(-1, -1, h * w)
        hist_feat = hist_feat.permute(0, 2, 1)
        tokens = feat.flatten(2).permute(0, 2, 1)
        fused, _ = self.cross_attn(tokens, hist_feat, hist_feat)
        fused_map = torch.cat([feat, fused.permute(0, 2, 1).reshape(b, 32, h, w)], dim=1)
        grid = self.head(fused_map)
        aff = grid[:, : self.cfg.affine_dim].view(b, self.cfg.affine_dim, h, w)
        alpha = F.softplus(grid[:, self.cfg.affine_dim : self.cfg.affine_dim + 1]).view(b, 1, h, w)
        styled = slice_bilateral_grid(
            aff, alpha, content, self.cfg.grid_guidance, self.cfg.grid_height, self.cfg.grid_width
        )
        return self.lut(styled.clamp(0, 1))
