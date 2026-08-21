"""Synthetic VoD-style calibration batch for smoke training."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.clrnet.config import CLRNetConfig
from ltx_trainer.clrnet.equirect import stack_lidar_depth, stack_radar_depth
from ltx_trainer.clrnet.se3 import quat_normalize, transform_from_qt


def _random_quat(batch: int, device: torch.device) -> Tensor:
    q = torch.randn(batch, 4, device=device)
    return quat_normalize(q)


def _random_transform(batch: int, device: torch.device, scale: float = 0.1) -> tuple[Tensor, Tensor, Tensor]:
    q = _random_quat(batch, device)
    t = torch.randn(batch, 3, device=device) * scale
    return q, t, transform_from_qt(q, t)


def synthetic_batch(
    cfg: CLRNetConfig,
    batch_size: int = 2,
    *,
    device: torch.device | str | None = None,
    num_points: int = 128,
) -> dict[str, Tensor]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    h, w = cfg.height, cfg.width
    image = torch.rand(batch_size, 3, h, w, device=dev)
    depth_pred = torch.rand(batch_size, 1, h, w, device=dev)

    pts = torch.randn(batch_size, num_points, 3, device=dev)
    pts[..., 2] = pts[..., 2].abs() + 1.0
    intensity = torch.rand(batch_size, num_points, device=dev)
    rcs = torch.rand(batch_size, num_points, device=dev)
    vel = torch.randn(batch_size, num_points, device=dev) * 0.1
    time_ch = torch.rand(batch_size, num_points, device=dev)

    lidar_depth = stack_lidar_depth(pts, intensity, height=h, width=w)
    radar_depth = stack_radar_depth(pts, rcs, vel, time_ch, height=h, width=w)

    q_cl_gt, t_cl_gt, _ = _random_transform(batch_size, dev, 0.05)
    q_lr_gt, t_lr_gt, _ = _random_transform(batch_size, dev, 0.05)
    q_rc_gt, t_rc_gt, _ = _random_transform(batch_size, dev, 0.05)

    q_cl, t_cl, _ = _random_transform(batch_size, dev, 0.08)
    q_lr, t_lr, _ = _random_transform(batch_size, dev, 0.08)
    q_rc, t_rc, _ = _random_transform(batch_size, dev, 0.08)

    return {
        "image": image,
        "depth_pred": depth_pred,
        "lidar_depth": lidar_depth,
        "radar_depth": radar_depth,
        "points": pts,
        "q_cl": q_cl,
        "t_cl": t_cl,
        "q_lr": q_lr,
        "t_lr": t_lr,
        "q_rc": q_rc,
        "t_rc": t_rc,
        "q_cl_gt": q_cl_gt,
        "t_cl_gt": t_cl_gt,
        "q_lr_gt": q_lr_gt,
        "t_lr_gt": t_lr_gt,
        "q_rc_gt": q_rc_gt,
        "t_rc_gt": t_rc_gt,
    }
