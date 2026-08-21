"""Semantic Gaussian optimization (Sec. II-B, Eq. 3–5)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig
from ltx_trainer.panogsdet.gaussian_lift import SemanticGaussianState


class VoxelSemanticRefine(nn.Module):
    """Dense voxel stub for sparse SubMConv3d semantic refinement (Eq. 3)."""

    def __init__(self, feature_dim: int, num_classes: int, voxel_size: float) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.conv = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, 5, padding=2),
            nn.GroupNorm(1, feature_dim),
            nn.GELU(),
            nn.Conv3d(feature_dim, feature_dim, 5, padding=2),
            nn.GELU(),
        )
        self.cls_conv = nn.Conv3d(feature_dim, num_classes, 1)

    def _voxelize(self, centers: Tensor, features: Tensor) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
        b, n, c = features.shape
        vox = self.voxel_size
        mins = centers.min(dim=1).values
        idx = ((centers - mins.unsqueeze(1)) / vox).long().clamp(min=0)
        gx = idx[..., 0].max().item() + 1
        gy = idx[..., 1].max().item() + 1
        gz = idx[..., 2].max().item() + 1
        gx = gy = gz = max(4, min(int(max(gx, gy, gz)), 32))
        vol = torch.zeros(b, c, gx, gy, gz, device=features.device, dtype=features.dtype)
        counts = torch.zeros(b, 1, gx, gy, gz, device=features.device, dtype=features.dtype)
        cap = min(n, 512)
        for bi in range(b):
            for ni in range(cap):
                ix = idx[bi, ni, 0].item() % gx
                iy = idx[bi, ni, 1].item() % gy
                iz = idx[bi, ni, 2].item() % gz
                vol[bi, :, ix, iy, iz] += features[bi, ni]
                counts[bi, 0, ix, iy, iz] += 1.0
        vol = vol / counts.clamp(min=1.0)
        return vol, idx, (gx, gy, gz)

    def forward(self, state: SemanticGaussianState) -> SemanticGaussianState:
        vol, idx, (gx, gy, gz) = self._voxelize(state.centers, state.features)
        refined = self.conv(vol)
        cls_vol = self.cls_conv(refined)
        b, n, c = state.features.shape
        new_feat = state.features.clone()
        new_logits = state.category_logits.clone()
        cap = min(n, 512)
        for bi in range(b):
            for ni in range(cap):
                ix = idx[bi, ni, 0].item() % gx
                iy = idx[bi, ni, 1].item() % gy
                iz = idx[bi, ni, 2].item() % gz
                new_feat[bi, ni] = refined[bi, :, ix, iy, iz]
                logits_voxel = cls_vol[bi, :, ix, iy, iz]
                new_logits[bi, ni] = 0.5 * (state.category_logits[bi, ni] + logits_voxel)
        return SemanticGaussianState(
            state.centers,
            new_feat,
            state.scales,
            state.rotations,
            state.opacity,
            new_logits,
        )


class CenterRefineBlock(nn.Module):
    """Eq. (4): residual center offset."""

    def __init__(self, feature_dim: int, step_scale: float) -> None:
        super().__init__()
        self.step_scale = step_scale
        self.mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 3))

    def forward(self, state: SemanticGaussianState) -> SemanticGaussianState:
        delta = (torch.sigmoid(self.mlp(state.features)) - 0.5) * self.step_scale
        centers = state.centers + delta
        return SemanticGaussianState(
            centers, state.features, state.scales, state.rotations, state.opacity, state.category_logits
        )


class CovarianceRefineBlock(nn.Module):
    """Eq. (5): scale and rotation residual refinement."""

    def __init__(self, feature_dim: int, beta: float, eta: float) -> None:
        super().__init__()
        self.beta = beta
        self.eta = eta
        self.scale_mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(), nn.Linear(feature_dim // 2, 3))
        self.rot_mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(), nn.Linear(feature_dim // 2, 3))

    def forward(self, state: SemanticGaussianState) -> SemanticGaussianState:
        sci = (torch.sigmoid(self.scale_mlp(state.features)) - 0.5) * self.beta
        roi = (torch.sigmoid(self.rot_mlp(state.features)) - 0.5) * self.eta
        scales = (state.scales + sci) / 2.0
        rotations = (state.rotations + roi) / 2.0
        return SemanticGaussianState(
            state.centers, state.features, scales, rotations, state.opacity, state.category_logits
        )


class SemanticGaussianOptimizer(nn.Module):
    def __init__(self, cfg: PanoGSDetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoGSDetConfig()
        self.blocks = nn.ModuleList()
        for _ in range(self.cfg.opt_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "semantic": VoxelSemanticRefine(
                            self.cfg.feature_dim, self.cfg.num_classes, self.cfg.voxel_size
                        ),
                        "center": CenterRefineBlock(self.cfg.feature_dim, self.cfg.center_step_scale),
                        "cov": CovarianceRefineBlock(
                            self.cfg.feature_dim, self.cfg.cov_beta, self.cfg.cov_eta
                        ),
                    }
                )
            )

    def forward(self, state: SemanticGaussianState) -> SemanticGaussianState:
        for block in self.blocks:
            state = block["semantic"](state)
            state = block["center"](state)
            state = block["cov"](state)
        return state
