"""Canonical Viewing Space + yaw-centered ERP (Sec. 3.2, 4)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.gimbal360.topology import roll_azimuth


def center_erp_by_yaw(erp: Tensor, yaw_index: int) -> Tensor:
    """
    I_centered = Roll(I_ERP, −ψ_input) — perspective always at canonical center.
    yaw_index: pixel offset along width corresponding to input heading.
    """
    return roll_azimuth(erp, -yaw_index)


def gravity_aligned_horizon_mask(height: int, width: int, *, device: torch.device | None = None) -> Tensor:
    """Binary mask: equator band (canonical horizon at v = H/2)."""
    dev = device or torch.device("cpu")
    v = torch.arange(height, device=dev, dtype=torch.float32)
    equator = height / 2.0
    band = (v - equator).abs() < max(height * 0.05, 2.0)
    return band.unsqueeze(1).expand(height, width).float()
