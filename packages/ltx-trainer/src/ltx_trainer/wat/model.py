"""Shared token-wise WAT autoencoder stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.wat.config import WATConfig
from ltx_trainer.wat.schema import TokenBatch


class WATSharedStub(nn.Module):
    """Linear modality adapters + LayerNorm-MLP encoder/decoder per token."""

    def __init__(self, cfg: WATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WATConfig()
        d = self.cfg.token_width
        h = self.cfg.hidden_dim
        z = self.cfg.latent_dim
        self.adapters_in = nn.ModuleDict(
            {
                "audio": nn.Linear(1, d),
                "image": nn.Linear(3, d),
                "video": nn.Linear(3, d),
            }
        )
        self.adapters_out = nn.ModuleDict(
            {
                "audio": nn.Linear(d, 1),
                "image": nn.Linear(d, 3),
                "video": nn.Linear(d, 3),
            }
        )
        self.mod_emb = nn.Embedding(3, d)
        self.rank_emb = nn.Embedding(4, d)
        self.scale_emb = nn.Embedding(4, d)
        self.subband_emb = nn.Embedding(16, d)
        self.pos_proj = nn.Linear(3, d)
        self.enc = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, z),
        )
        self.dec = nn.Sequential(
            nn.LayerNorm(z),
            nn.Linear(z, h),
            nn.GELU(),
            nn.Linear(h, d),
        )
        self.scales = {
            "audio": self.cfg.audio_scale,
            "image": self.cfg.image_scale,
            "video": self.cfg.video_scale,
        }

    def forward_tokens(
        self,
        batch: TokenBatch,
        modality: str,
        *,
        use_metadata: bool = True,
    ) -> Tensor:
        s = self.scales[modality]
        v = batch.values * s
        h = self.adapters_in[modality](v)
        if use_metadata:
            h = (
                h
                + self.mod_emb(batch.modality)
                + self.rank_emb(batch.rank)
                + self.scale_emb(batch.scale)
                + self.subband_emb(batch.subband)
                + self.pos_proj(batch.position)
            )
        z = self.enc(h)
        h2 = self.dec(z)
        return self.adapters_out[modality](h2) / s
