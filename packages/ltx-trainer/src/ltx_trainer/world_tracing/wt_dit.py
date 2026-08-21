"""WT-DiT flow-matching diffusion transformer stub."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.world_tracing.attention import AdaLNTime, LayerFiLM, PreNormAttention, WTDiTBlock
from ltx_trainer.world_tracing.config import WTConfig
from ltx_trainer.world_tracing.moge_encoder import MoGeEncoderStub


class WTDiT(nn.Module):
    """Pixel-aligned multilayer geometry flow matcher (Sec. 3.2)."""

    def __init__(self, cfg: WTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        self.ph = cfg.height // p
        self.pw = cfg.width // p
        self.num_patches = self.ph * self.pw
        geo_in = 3 * p * p
        self.encoder = MoGeEncoderStub(in_ch=4, feat_dim=cfg.moge_feat_dim)
        self.trainable_proj = nn.Conv2d(cfg.moge_feat_dim, cfg.width_dim, 1)
        self.geo_proj = nn.Linear(geo_in + cfg.width_dim, cfg.width_dim)
        self.layer_film = LayerFiLM(cfg.num_layers, cfg.width_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, cfg.width_dim),
            nn.SiLU(),
            nn.Linear(cfg.width_dim, cfg.width_dim),
        )
        self.layer_emb = nn.Embedding(cfg.num_layers, cfg.width_dim)
        self.adaln = AdaLNTime(cfg.width_dim)
        self.blocks = nn.ModuleList([WTDiTBlock(cfg.width_dim, cfg.num_heads) for _ in range(cfg.num_blocks)])
        self.head = nn.Linear(cfg.width_dim, 3 * p * p)

    def patchify(self, x: Tensor) -> Tensor:
        """(B, L, H, W, 3) → (B, L, P, patch_dim)."""
        b, layers, h, w, c = x.shape
        p = self.cfg.patch_size
        x = x.view(b, layers, self.ph, p, self.pw, p, c)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, layers, self.num_patches, p * p * c)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        """(B, L, P, 3*p*p) → (B, L, H, W, 3)."""
        b, layers, patches, _ = tokens.shape
        p = self.cfg.patch_size
        x = tokens.view(b, layers, self.ph, self.pw, p, p, 3)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(b, layers, self.ph * p, self.pw * p, 3)

    def img_to_patch_tokens(self, img: Tensor) -> Tensor:
        b, c, h, w = img.shape
        p = self.cfg.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        return x.mean(dim=(-1, -2)).permute(0, 2, 3, 1).reshape(b, self.num_patches, c)

    def fuse_tokens(self, geo: Tensor, img_feat: Tensor) -> Tensor:
        b, layers, patches, _ = geo.shape
        img = self.img_to_patch_tokens(img_feat)
        img = img.unsqueeze(1).expand(b, layers, patches, img.shape[-1])
        fused = torch.cat([geo, img], dim=-1)
        return self.geo_proj(fused)

    def forward(self, rgba: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            rgba: (B, 4, H, W)
            xt: (B, L, H, W, 3) noisy geometry
            t: (B, L) or (B,) diffusion time in [0, 1]
        Returns:
            pred_x0: (B, L, H, W, 3)
        """
        b = rgba.shape[0]
        img = self.trainable_proj(self.encoder(rgba))
        geo = self.patchify(xt)
        tokens = self.fuse_tokens(geo, img)
        tokens = self.layer_film(tokens)

        if t.ndim == 1:
            t = t.unsqueeze(1).expand(b, self.cfg.num_layers)
        t_mean = t.mean(dim=1, keepdim=True)
        t_emb = self.time_emb(t_mean)
        layer_ids = torch.arange(self.cfg.num_layers, device=rgba.device)
        tokens = tokens + self.layer_emb(layer_ids).view(1, self.cfg.num_layers, 1, -1)
        tokens = self.adaln(tokens, t_emb)

        for block in self.blocks:
            tokens = block(tokens)

        patch_xyz = self.head(tokens)
        return self.unpatchify(patch_xyz)
