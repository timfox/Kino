"""BEV-based occupancy VAE (Sec. 3.2, Fig. 3a)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.anyscene.config import AnySceneConfig


def occupancy_to_bev_feature(labels: Tensor, class_embed: nn.Embedding) -> Tensor:
    """Height-stack class embeddings: (B, H, W, Z) → (B, C', H, W)."""
    b, h, w, z = labels.shape
    idx = labels.long().clamp(0, class_embed.num_embeddings - 1)
    emb = class_embed(idx)
    b, h, w, z, d = emb.shape
    return emb.permute(0, 3, 4, 1, 2).reshape(b, z * d, h, w)


class OccupancyVAE(nn.Module):
    """Compact 2D encoder/decoder on BEV-stacked occupancy."""

    def __init__(self, cfg: AnySceneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        zc = cfg.occupancy_classes
        d = cfg.class_embed_dim
        self.class_embed = nn.Embedding(zc, d)
        in_ch = d * cfg.occupancy_z
        lc = cfg.latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, lc * 2, 3, stride=1, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(lc, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, in_ch, 3, padding=1),
        )

    def encode(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        x = occupancy_to_bev_feature(labels, self.class_embed)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return z, mu, logvar, recon


def focal_loss_logits(logits: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    """Focal loss on per-voxel class logits."""
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def vae_loss(
    recon: Tensor,
    labels: Tensor,
    mu: Tensor,
    logvar: Tensor,
    cfg: AnySceneConfig,
) -> Tensor:
    """Eq. (2): focal + KL (Lovász-Softmax omitted in smoke path)."""
    b, _, h, w = recon.shape
    z = labels.shape[-1]
    n_cls = cfg.occupancy_classes
    logits = recon[:, : n_cls * z].reshape(b, n_cls, z, h, w).permute(0, 3, 4, 2, 1).reshape(-1, n_cls)
    target = labels.reshape(-1)
    lf = focal_loss_logits(logits, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return lf + cfg.kl_weight * kl
