"""CLRNet, CRNet, and CLRNet+4 stubs (Fig. 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.clrnet.config import CLRNetConfig
from ltx_trainer.clrnet.correlation import CorrelationLayer
from ltx_trainer.clrnet.encoders import CameraEncoder, DepthImageEncoder
from ltx_trainer.clrnet.regression import ExtrinsicRegressionHead, SharedFeatureRegression


class CLRNet(nn.Module):
    """Joint camera–lidar–radar calibration network."""

    def __init__(self, cfg: CLRNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CLRNetConfig()
        c = self.cfg
        self.camera = CameraEncoder(c)
        self.lidar = DepthImageEncoder(c.lidar_channels, c)
        self.radar = DepthImageEncoder(c.radar_channels, c)
        self.corr = CorrelationLayer()
        corr_dim = 32 * 4 * 8  # flatten size from CorrelationLayer post
        self.regression = SharedFeatureRegression(corr_dim)
        self._corr_dim = corr_dim

    def encode(
        self,
        image: Tensor,
        lidar_depth: Tensor,
        radar_depth: Tensor,
        depth_pred: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.camera(image, depth_pred),
            self.lidar(lidar_depth),
            self.radar(radar_depth),
        )

    def forward(
        self,
        image: Tensor,
        lidar_depth: Tensor,
        radar_depth: Tensor,
        *,
        depth_pred: Tensor | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        fc, fl, fr = self.encode(image, lidar_depth, radar_depth, depth_pred)
        corr_cl = self.corr(fc, fl)
        corr_lr = self.corr(fl, fr)
        corr_rc = self.corr(fr, fc)
        return self.regression(corr_cl, corr_lr, corr_rc)


class CRNet(nn.Module):
    """Pairwise camera–radar variant (Sec. III-E)."""

    def __init__(self, cfg: CLRNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CLRNetConfig()
        c = self.cfg
        self.camera = CameraEncoder(c)
        self.radar = DepthImageEncoder(c.radar_channels, c)
        self.corr = CorrelationLayer()
        self.head = ExtrinsicRegressionHead(32 * 4 * 8)

    def forward(
        self,
        image: Tensor,
        radar_depth: Tensor,
        *,
        depth_pred: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        fc = self.camera(image, depth_pred)
        fr = self.radar(radar_depth)
        feat = self.corr(fc, fr)
        return self.head(feat)


class CLRNetPlus4(CLRNet):
    """Four-frame rigid-platform variant (Sec. III-F)."""

    def __init__(self, cfg: CLRNetConfig | None = None) -> None:
        cfg = cfg or CLRNetConfig()
        cfg = CLRNetConfig(**{**cfg.__dict__, "num_frames": 4})
        super().__init__(cfg)
        self.frame_fuse_cam = nn.Conv2d(cfg.feature_dim * 4, cfg.feature_dim, 1)
        self.frame_fuse_lid = nn.Conv2d(cfg.feature_dim * 4, cfg.feature_dim, 1)
        self.frame_fuse_rad = nn.Conv2d(cfg.feature_dim * 4, cfg.feature_dim, 1)

    def encode_multiframe(
        self,
        image: Tensor,
        lidar_depth: Tensor,
        radar_depth: Tensor,
        depth_pred: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Inputs [B, T, C, H, W]."""
        maps_c, maps_l, maps_r = [], [], []
        t = image.shape[1]
        for i in range(t):
            dp = depth_pred[:, i] if depth_pred is not None else None
            fc, fl, fr = self.encode(image[:, i], lidar_depth[:, i], radar_depth[:, i], dp)
            maps_c.append(fc)
            maps_l.append(fl)
            maps_r.append(fr)
        return (
            self.frame_fuse_cam(torch.cat(maps_c, dim=1)),
            self.frame_fuse_lid(torch.cat(maps_l, dim=1)),
            self.frame_fuse_rad(torch.cat(maps_r, dim=1)),
        )

    def forward(
        self,
        image: Tensor,
        lidar_depth: Tensor,
        radar_depth: Tensor,
        *,
        depth_pred: Tensor | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        if image.dim() == 5:
            fc, fl, fr = self.encode_multiframe(image, lidar_depth, radar_depth, depth_pred)
            corr_cl = self.corr(fc, fl)
            corr_lr = self.corr(fl, fr)
            corr_rc = self.corr(fr, fc)
            return self.regression(corr_cl, corr_lr, corr_rc)
        return super().forward(image, lidar_depth, radar_depth, depth_pred=depth_pred)
