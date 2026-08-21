"""Generative registration pipeline stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.genpcr.config import GenPcrConfig
from ltx_trainer.genpcr.fusion import fuse_geo_color, xyz_rgb_point_cloud
from ltx_trainer.genpcr.match_controlnet import MatchControlNetStub


class GeometryEncoderStub(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, pts: Tensor) -> Tensor:
        return self.mlp(pts)


class ColorEncoderStub(nn.Module):
    """DINOv2/SD placeholder: conv on per-point random colors from image sample."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Linear(3, dim)

    def forward(self, pts: Tensor, image: Tensor) -> Tensor:
        n = pts.shape[0]
        h, w = image.shape[-2], image.shape[-1]
        u = (pts[:, 0].tanh() * 0.5 + 0.5) * (w - 1)
        v = (pts[:, 1].tanh() * 0.5 + 0.5) * (h - 1)
        ui = u.long().clamp(0, w - 1)
        vi = v.long().clamp(0, h - 1)
        img = image.squeeze(0) if image.dim() == 4 else image
        return self.mlp(img[:, vi, ui].t())


class GenerativePcrStub(nn.Module):
    def __init__(self, cfg: GenPcrConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GenPcrConfig()
        self.gen = MatchControlNetStub(self.cfg)
        self.geo_enc = GeometryEncoderStub(self.cfg.geo_dim)
        self.rgb_enc = ColorEncoderStub(self.cfg.color_dim)

    def enhanced_descriptors(
        self,
        points: Tensor,
        image: Tensor,
    ) -> Tensor:
        f_geo = self.geo_enc(points)
        f_rgb = self.rgb_enc(points, image)
        return fuse_geo_color(
            f_geo,
            f_rgb,
            omega=self.cfg.fusion_weight,
            rgb_dim=self.cfg.geo_dim,
        )

    def forward(
        self,
        points_p: Tensor,
        points_q: Tensor,
    ) -> dict[str, Tensor]:
        gen = self.gen(points_p, points_q)
        desc_p = self.enhanced_descriptors(points_p, gen["image_p"])
        desc_q = self.enhanced_descriptors(points_q, gen["image_q"])
        # correspondence score proxy: negative L2 between mean descriptors
        mp, mq = desc_p.mean(0), desc_q.mean(0)
        match_cost = torch.norm(mp - mq, p=2)
        h, w = self.cfg.image_height, self.cfg.image_width
        return {
            **gen,
            "desc_p": desc_p,
            "desc_q": desc_q,
            "match_cost": match_cost,
            "color_pcd_p": xyz_rgb_point_cloud(
                points_p, gen["image_p"], height=h, width=w
            ),
            "color_pcd_q": xyz_rgb_point_cloud(
                points_q, gen["image_q"], height=h, width=w
            ),
        }
