"""Synthetic multi-session ground texture with tape wear."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.ground_texture_slam.schema import Pose2D


def _carpet_texture(h: int, w: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, h, w, generator=g) * 0.15 + 0.35
    noise = torch.randn(3, h, w, generator=g) * 0.03
    return (base + noise).clamp(0, 1)


def apply_wear(image: Tensor, session: int, *, severity: float | None = None) -> Tensor:
    """Simulate tape wear increasing across sessions (Fig. 2)."""
    sev = severity if severity is not None else session / 4.0
    h, w = image.shape[-2:]
    wear = torch.zeros(1, h, w)
    g = torch.Generator().manual_seed(session + 7)
    mask = torch.rand(1, h, w, generator=g) < sev * 0.45
    wear = mask.float() * 0.25
    out = image * (1.0 - wear) + wear
    return out.clamp(0, 1)


def session_poses(session: int, *, n: int = 40, radius: float = 2.0) -> list[Pose2D]:
    """Similar but not identical paths per session (Fig. 3)."""
    poses: list[Pose2D] = []
    offset = session * 0.05
    for t in range(n):
        ang = 2 * math.pi * t / n + offset
        poses.append(Pose2D(radius * math.cos(ang), radius * math.sin(ang), ang + offset, session, t))
    return poses


def synthesize_session(
    session: int,
    *,
    size: int = 64,
    n_poses: int = 40,
) -> tuple[list[Tensor], list[Pose2D]]:
    """Return images and GT poses for one session."""
    base = _carpet_texture(size, size, seed=session)
    images = [apply_wear(base.clone(), session) for _ in range(n_poses)]
    return images, session_poses(session, n=n_poses)
