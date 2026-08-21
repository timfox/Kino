"""SphereFusion full network stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.spherefusion.config import SphereFusionConfig
from ltx_trainer.spherefusion.gatefuse import BiFuseStub, GateFuse, UniFuseStub
from ltx_trainer.spherefusion.mesh_ops import MeshConvStub, MeshPoolStub, faf_neighbors
from ltx_trainer.spherefusion.projection import e2s_sample


class ImageEncoderStub(nn.Module):
    def __init__(self, cfg: SphereFusionConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(self.stem(x))


class MeshEncoderStub(nn.Module):
    def __init__(self, cfg: SphereFusionConfig, n_tri: int) -> None:
        super().__init__()
        self.conv = MeshConvStub(64)
        self.pool = MeshPoolStub()
        self.n_tri = n_tri
        self.mr = cfg.mesh_mr
        self.cache = cfg.faf_cache

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = min(self.n_tri, h * w)
        f = x.view(b, c, -1).transpose(1, 2)[:, :n, :64]
        if f.shape[-1] < 64:
            f = torch.nn.functional.pad(f, (0, 64 - f.shape[-1]))
        nbr = faf_neighbors(n, mr=self.mr, use_cache=self.cache)
        f = self.conv(f, nbr)
        return self.pool(f)


class SphereFusionStub(nn.Module):
    def __init__(self, cfg: SphereFusionConfig | None = None, *, fusion: str = "gatefuse") -> None:
        super().__init__()
        self.cfg = cfg or SphereFusionConfig()
        self.img_enc = ImageEncoderStub(self.cfg)
        self.mesh_enc = MeshEncoderStub(self.cfg, n_tri=256)
        ch = 64
        if fusion == "bifuse":
            self.fuse = BiFuseStub()
        elif fusion == "unifuse":
            self.fuse = UniFuseStub()
        else:
            self.fuse = GateFuse(ch)
        self.depth_head = nn.Sequential(
            nn.Linear(ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, erp: Tensor) -> dict[str, Tensor]:
        feq = self.img_enc(erp)
        fsp = self.mesh_enc(erp)
        feq_s = e2s_sample(feq, fsp.shape[1])
        fused = self.fuse(fsp, feq_s)
        depth_mesh = self.depth_head(fused).squeeze(-1)
        b, n = depth_mesh.shape
        gh = max(1, int(n**0.5))
        gw = max(1, (n + gh - 1) // gh)
        if gh * gw > n:
            depth_mesh = torch.nn.functional.pad(depth_mesh, (0, gh * gw - n))
        depth_grid = depth_mesh.view(b, 1, gh, gw)
        depth_erp = torch.nn.functional.interpolate(
            depth_grid, size=erp.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"depth": depth_erp.squeeze(1), "depth_mesh": depth_mesh[:, :n]}
