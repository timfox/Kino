"""Transition-plane view matrices (Eq. 6–7, Sec. III-B)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.tpgs.config import TRANSITION_YAW_DEG


def rot_x(psi_deg: float, device: torch.device | None = None) -> Tensor:
    p = math.radians(psi_deg)
    c, s = math.cos(p), math.sin(p)
    return torch.tensor(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=torch.float32,
        device=device,
    )


def rot_y(psi_deg: float, device: torch.device | None = None) -> Tensor:
    p = math.radians(psi_deg)
    c, s = math.cos(p), math.sin(p)
    return torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=torch.float32,
        device=device,
    )


def cube_face_yaws() -> list[float]:
    """Horizontal faces: 0°, 90°, 180°, 270°; vertical: ±90° as elevation stubs."""
    return [0.0, 90.0, 180.0, 270.0, -90.0, 90.0]


def view_rotations(
    *,
    include_transition: bool = True,
    device: torch.device | None = None,
) -> list[tuple[str, Tensor]]:
    """
    Return (label, 3×3 rotation) for six cube faces and six transition planes.
    Transition uses R_y(45°) on top of face yaw (Eq. 7).
    """
    views: list[tuple[str, Tensor]] = []
    horiz = [("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", 270.0)]
    vert = [("down", -90.0), ("up", 90.0)]
    for name, yaw in horiz:
        r = rot_y(yaw, device=device)
        views.append((f"cube_{name}", r))
        if include_transition:
            views.append((f"tp_{name}", r @ rot_y(TRANSITION_YAW_DEG, device=device)))
    for name, pitch in vert:
        r = rot_x(pitch, device=device)
        views.append((f"cube_{name}", r))
        if include_transition:
            views.append((f"tp_{name}", r @ rot_y(TRANSITION_YAW_DEG, device=device)))
    return views
