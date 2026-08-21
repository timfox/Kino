"""Pitch/yaw gaze and head pose utilities (Sec. 2.1–2.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pitch_yaw_to_direction(pitch: Tensor, yaw: Tensor) -> Tensor:
    """Unit gaze/head direction from pitch and yaw (radians). Shape (..., 2) -> (..., 3)."""
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    x = cp * sy
    y = sp
    z = cp * cy
    return torch.stack([x, y, z], dim=-1)


def directions_from_pose(pose: Tensor) -> Tensor:
    """pose: (..., 2) as [pitch, yaw] in radians."""
    return pitch_yaw_to_direction(pose[..., 0], pose[..., 1])


def angular_error_deg(pred: Tensor, target: Tensor) -> Tensor:
    """Mean 3D angular error in degrees between pitch/yaw sequences.

    pred, target: (T, 2) or (B, T, 2) in radians.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    dp = directions_from_pose(pred)
    dt = directions_from_pose(target)
    cos = (dp * dt).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)
    return ang.mean(dim=-1)


def deg2rad(deg: float | Tensor) -> float | Tensor:
    if isinstance(deg, Tensor):
        return deg * (math.pi / 180.0)
    return deg * math.pi / 180.0
