"""Modality-specific encoders (Sec. III-C.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.clrnet.config import CLRNetConfig


def _resnet_stub(in_ch: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 7, stride=2, padding=3),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(64, out_dim, 3, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.AdaptiveAvgPool2d((16, 32)),
    )


class CameraEncoder(nn.Module):
    def __init__(self, cfg: CLRNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rgb = _resnet_stub(3, cfg.feature_dim)
        self.depth_enc = _resnet_stub(1, cfg.feature_dim) if cfg.use_depth_branch else None
        in_fuse = cfg.feature_dim * (2 if cfg.use_depth_branch else 1)
        self.fuse = nn.Conv2d(in_fuse, cfg.feature_dim, 1)

    def forward(self, image: Tensor, depth_pred: Tensor | None = None) -> Tensor:
        feats = [self.rgb(image)]
        if self.depth_enc is not None and depth_pred is not None:
            feats.append(self.depth_enc(depth_pred))
        x = feats[0] if len(feats) == 1 else self.fuse(torch.cat(feats, dim=1))
        return x


class DepthImageEncoder(nn.Module):
    """Lidar (2ch) or radar (4ch) ERP depth encoder."""

    def __init__(self, in_channels: int, cfg: CLRNetConfig) -> None:
        super().__init__()
        self.net = _resnet_stub(in_channels, cfg.feature_dim)

    def forward(self, depth_image: Tensor) -> Tensor:
        return self.net(depth_image)


class MultiFrameEncoder(nn.Module):
    """Concatenate per-frame feature maps along channels (CLRNet+4, Sec. III-F)."""

    def __init__(self, inner: nn.Module, num_frames: int, out_dim: int) -> None:
        super().__init__()
        self.inner = inner
        self.num_frames = num_frames
        self.compress = nn.Conv2d(out_dim * num_frames, out_dim, 1)

    def forward(self, *args: Tensor) -> Tensor:
        # args[0] may be [B,T,C,H,W] or list handled by caller
        x = args[0]
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            maps = [self.inner(x[:, i]) for i in range(t)]
            return self.compress(torch.cat(maps, dim=1))
        return self.inner(x)
