"""Horizon360 perspective sampling — mixture distribution (Sec. 4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.gimbal360.config import Gimbal360Config


def sample_pose_mixture(
    batch_size: int,
    cfg: Gimbal360Config,
    *,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """
    P(Θ) = λ·N(μ_real, σ²) + (1−λ)·U(Θ_min, Θ_max) for pitch, roll, FOV.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    lam = cfg.mixture_lambda
    pitch = torch.empty(batch_size, device=dev)
    roll = torch.empty(batch_size, device=dev)
    fov_deg = torch.empty(batch_size, device=dev)
    for i in range(batch_size):
        if torch.rand(1, device=dev).item() < lam:
            pitch[i] = torch.randn(1, device=dev).item() * math.radians(15.0)
        else:
            pitch[i] = (torch.rand(1, device=dev).item() * 2 - 1) * math.radians(45.0)
        if torch.rand(1, device=dev).item() < 0.8:
            roll[i] = torch.randn(1, device=dev).item() * math.radians(5.0)
        else:
            roll[i] = (torch.rand(1, device=dev).item() * 2 - 1) * math.radians(45.0)
        if torch.rand(1, device=dev).item() < 0.5:
            fov_deg[i] = 60.0 + torch.randn(1, device=dev).item() * 10.0
        else:
            fov_deg[i] = 45.0 + torch.rand(1, device=dev).item() * 55.0
    yaw = (torch.rand(batch_size, device=dev) * 2 - 1) * math.pi
    return {
        "pitch_rad": pitch,
        "roll_rad": roll,
        "yaw_rad": yaw,
        "fov_deg": fov_deg,
    }


def focal_from_vertical_fov(fov_deg: Tensor, aspect_ratio: float = 4 / 3) -> tuple[Tensor, Tensor]:
    """fy = 1/tan(θ_v/2), fx = fy/r (Sec. 4)."""
    theta_v = fov_deg * math.pi / 180.0
    fy = 1.0 / torch.tan(theta_v / 2.0 + 1e-8)
    fx = fy / aspect_ratio
    return fx, fy
