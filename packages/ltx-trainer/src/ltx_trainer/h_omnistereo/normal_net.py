"""Equirectangular heading-aligned monocular normal estimator (Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
from ltx_trainer.h_omnistereo.heading_normal import camera_to_heading_aligned, longitude_grid
from ltx_trainer.h_omnistereo.ray_embed import RayEmbedding


class RayCrossAttention(nn.Module):
    """Fuse visual features with ray embeddings (keys/values = rays, queries = features)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, feat: Tensor, ray: Tensor) -> Tensor:
        q, k, v = self.q(feat), self.k(ray), self.v(ray)
        b, c, h, w = q.shape
        attn = torch.softmax((q * k).sum(dim=1, keepdim=True) / (c**0.5 + 1e-6), dim=1)
        out = feat + self.proj(attn * v)
        return out


class HeadingAlignedNormalNet(nn.Module):
    """
    ViT-scale stub: conv encoder + ray cross-attention + FPN upsample → unit normals in HA frame.
    """

    def __init__(self, cfg: HOmniStereoConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HOmniStereoConfig()
        d = self.cfg.feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, d, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ray_embed = RayEmbedding(dim=d)
        self.fuse = RayCrossAttention(d)
        self.refine = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(d, 3, 1)
        self._prior_features: Tensor | None = None

    def encode_priors(self, rgb: Tensor) -> Tensor:
        """Latent features before normal head (for side-tuning adapter)."""
        b, _, h, w = rgb.shape
        x = self.stem(rgb)
        ray = torch.nn.functional.interpolate(
            self.ray_embed(h, w, device=rgb.device),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.fuse(x, ray)
        x = self.refine(x)
        self._prior_features = x
        return x

    def forward(self, rgb: Tensor, *, return_camera_frame: bool = False) -> Tensor:
        x = self.encode_priors(rgb)
        n_cam = torch.nn.functional.normalize(self.head(x), dim=1, eps=1e-6)
        n_cam = torch.nn.functional.interpolate(n_cam, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        alpha = longitude_grid(rgb.shape[-1], rgb.shape[-2], device=rgb.device)
        n_ha = camera_to_heading_aligned(n_cam, alpha.unsqueeze(0).expand(rgb.shape[0], -1, -1))
        if return_camera_frame:
            return n_ha, n_cam
        return n_ha
