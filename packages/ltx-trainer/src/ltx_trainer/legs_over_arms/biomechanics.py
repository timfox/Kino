"""Derived biomechanical cues C (Sec. III-A)."""

from __future__ import annotations

import torch
from torch import Tensor


def leg_articulation_angles(kp: Tensor) -> Tensor:
    """Hip–knee–ankle angles from lower-body 3D keypoints (B, T, 10, 3)."""
    if kp.shape[-2] < 4:
        return torch.zeros(kp.shape[0], kp.shape[1], 3, device=kp.device, dtype=kp.dtype)
    angles = []
    for i in range(3):
        v1 = kp[..., i + 1, :] - kp[..., i, :]
        v2 = kp[..., i + 2, :] - kp[..., i + 1, :]
        cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
        angles.append(torch.acos(cos.clamp(-1, 1)))
    return torch.stack(angles, dim=-1)


def step_length(kp: Tensor) -> Tensor:
    """Ankle displacement magnitude between consecutive frames."""
    if kp.shape[1] < 2:
        return torch.zeros(kp.shape[0], 1, device=kp.device, dtype=kp.dtype)
    ankle = kp[..., -1, :2]
    return (ankle[:, 1:] - ankle[:, :-1]).norm(dim=-1).mean(dim=-1, keepdim=True)


def head_orientation(kp_upper: Tensor) -> Tensor:
    """Yaw proxy from nose/ear keypoints (B, T, 3)."""
    if kp_upper.shape[-2] < 2:
        return torch.zeros(kp_upper.shape[0], kp_upper.shape[1], 1, device=kp_upper.device)
    return torch.atan2(
        kp_upper[..., 1, 0] - kp_upper[..., 0, 0],
        kp_upper[..., 1, 2] - kp_upper[..., 0, 2] + 1e-6,
    ).unsqueeze(-1)


def cues_lower_body_3d(kp: Tensor) -> Tensor:
    angles = leg_articulation_angles(kp)
    step = step_length(kp)
    if step.dim() == 2:
        step = step.unsqueeze(1).expand(-1, angles.shape[1], -1)
    return torch.cat([angles, step], dim=-1)
