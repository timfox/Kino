"""Sphere-to-planar-Aware Feature Encoder (SAFE)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.faor.atfm import ATFM
from ltx_trainer.faor.config import FaorConfig


class CrossAttentionStub(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, fln: Tensor, fat: Tensor) -> Tensor:
        q = self.q(fln)
        k, v = self.k(fat), self.v(fat)
        dk = q.shape[1] ** 0.5
        attn = torch.softmax((q * k).sum(dim=1, keepdim=True) / dk, dim=-1)
        return self.out(attn * v)


class SAFEBlock(nn.Module):
    def __init__(self, ch: int, *, use_md: bool, use_ms: bool) -> None:
        super().__init__()
        self.use_md = use_md
        self.use_ms = use_ms
        self.ln = nn.GroupNorm(8, ch)
        self.atfm = ATFM(ch)
        self.ca = CrossAttentionStub(ch)
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.SiLU(), nn.Conv2d(ch * 2, ch, 1))

    def forward(self, fl: Tensor, *, md: Tensor | None, ms: Tensor | None) -> Tensor:
        fln = self.ln(fl)
        fat = self.atfm(fln, md=md if self.use_md else None, ms=ms if self.use_ms else None)
        fca = self.ca(fln, fat)
        ffca = fl + fca
        return ffca + self.mlp(self.ln(ffca))


class SAFEEncoder(nn.Module):
    def __init__(self, cfg: FaorConfig) -> None:
        super().__init__()
        self.stem = nn.Conv2d(cfg.in_ch, cfg.latent_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                SAFEBlock(cfg.latent_dim, use_md=cfg.use_md, use_ms=cfg.use_ms)
                for _ in range(min(cfg.safe_blocks, 4))  # stub: 4 blocks not 36
            ]
        )

    def forward(self, x: Tensor, *, md: Tensor | None = None, ms: Tensor | None = None) -> Tensor:
        z = self.stem(x)
        for blk in self.blocks:
            z = blk(z, md=md, ms=ms)
        return z
