"""Multi-hot BEV layout encoding (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.anyscene.config import AnySceneConfig


class BEVLayoutEncoder(nn.Module):
    """Sum active class embeddings per cell; empty cells use learnable background."""

    def __init__(self, cfg: AnySceneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.bev_channels
        d = cfg.class_embed_dim
        self.class_embed = nn.Embedding(c, d)
        self.empty_embed = nn.Parameter(torch.zeros(d))
        self.down = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1),
        )

    def forward(self, layout: Tensor) -> Tensor:
        """``layout`` (B, C, H, W) multi-hot → tokens (B, S, D) with S = token_grid²."""
        if layout.dim() != 4:
            raise ValueError("layout must be (B, C, H, W)")
        b, c, h, w = layout.shape
        if c != self.cfg.bev_channels:
            raise ValueError(f"expected {self.cfg.bev_channels} channels, got {c}")
        # (B, H, W, D)
        emb = torch.einsum("bchw,cd->bhwd", layout.float(), self.class_embed.weight)
        active = layout.sum(dim=1, keepdim=True).clamp(max=1.0)
        empty = self.empty_embed.view(1, 1, 1, -1).expand(b, h, w, -1)
        feat = emb + (1.0 - active.permute(0, 2, 3, 1)) * empty
        feat = feat.permute(0, 3, 1, 2)
        feat = self.down(feat)
        _, d, th, tw = feat.shape
        return feat.flatten(2).transpose(1, 2)
