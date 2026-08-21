"""Compact ERP visual encoder stub (patch embed + shallow blocks)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld.config import PanoWorldConfig


class ERPVisualEncoder(nn.Module):
    """Patchify ERP RGB and produce H^(0) before SSCA."""

    def __init__(self, cfg: PanoWorldConfig) -> None:
        super().__init__()
        p = cfg.patch_size
        self.patch_embed = nn.Conv2d(3, cfg.hidden_dim, kernel_size=p, stride=p)
        self.blocks = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self._cfg = cfg

    def forward(self, rgb: Tensor) -> Tensor:
        """rgb [B,3,H,W] → tokens [B,N,d]."""
        x = self.patch_embed(rgb)
        b, d, gh, gw = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.blocks(tokens)
        return tokens
