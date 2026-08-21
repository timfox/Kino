"""Synthetic ERP clips for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.s3po.config import S3POConfig


def synthetic_erp_clip(cfg: S3POConfig, *, num_frames: int = 3) -> Tensor:
    """LR clip [1,T,3,H,W] with latitude-varying pattern."""
    h, w = cfg.lr_height, cfg.lr_width
    yy = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    frames = []
    for t in range(num_frames):
        phase = 0.2 * t
        img = (0.5 + 0.3 * torch.sin(xx * 6.28 + phase) * torch.cos(yy * 3.14)).clamp(0, 1)
        frames.append(img.unsqueeze(0).expand(3, -1, -1))
    return torch.stack(frames, dim=0).unsqueeze(0)


def synthetic_hr_gt(cfg: S3POConfig, *, num_frames: int = 3) -> Tensor:
    """HR ground truth [1,T,3,sH,sW]."""
    lr = synthetic_erp_clip(cfg, num_frames=num_frames)
    s = cfg.scale
    b, t, c, h, w = lr.shape
    flat = lr.view(b * t, c, h, w)
    hr = torch.nn.functional.interpolate(flat, scale_factor=s, mode="bicubic", align_corners=False)
    return hr.view(b, t, c, h * s, w * s)
