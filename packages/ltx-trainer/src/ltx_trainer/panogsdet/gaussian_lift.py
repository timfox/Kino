"""Semantic 3D Gaussian lifting (Sec. II-A, Eq. 2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig
from ltx_trainer.panogsdet.erp_geometry import depth_map_to_points, flatten_points


class SemanticGaussianState:
    """Per-Gaussian parameters after lifting / optimization."""

    def __init__(
        self,
        centers: Tensor,
        features: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacity: Tensor,
        category_logits: Tensor,
    ) -> None:
        self.centers = centers
        self.features = features
        self.scales = scales
        self.rotations = rotations
        self.opacity = opacity
        self.category_logits = category_logits

    @property
    def num_gaussians(self) -> int:
        return self.centers.shape[1]


class GaussianLiftingMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, r_max: float) -> None:
        super().__init__()
        self.r_max = r_max
        hidden = max(in_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.scale_head = nn.Linear(hidden, 3)
        self.rot_head = nn.Linear(hidden, 3)
        self.opacity_head = nn.Linear(hidden, 1)
        self.cls_head = nn.Linear(hidden, num_classes)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.net(feat)
        sig = torch.sigmoid
        scales = sig(self.scale_head(h)) * self.r_max
        rotations = (sig(self.rot_head(h)) - 0.5) * 2.0 * math.pi
        opacity = sig(self.opacity_head(h))
        logits = self.cls_head(h)
        return scales, rotations, opacity, logits


class SemanticGaussianLifting(nn.Module):
    def __init__(self, cfg: PanoGSDetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoGSDetConfig()
        self.mlp = GaussianLiftingMLP(self.cfg.feature_dim, self.cfg.num_classes, self.cfg.r_max)

    def forward(self, depth: Tensor, sem_feat: Tensor, *, max_points: int | None = 2048) -> SemanticGaussianState:
        points = depth_map_to_points(depth)
        b, h, w, _ = points.shape
        centers = flatten_points(points)
        feat = sem_feat.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        n = centers.shape[1]
        if max_points is not None and n > max_points:
            idx = torch.linspace(0, n - 1, max_points, device=centers.device).long()
            centers = centers[:, idx]
            feat = feat[:, idx]
        scales, rotations, opacity, logits = self.mlp(feat)
        return SemanticGaussianState(centers, feat, scales, rotations, opacity.squeeze(-1), logits)
