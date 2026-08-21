"""Spherical epipolar geometry (Sec. 3.3, Eq. 6–9)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.campvg.erp_geometry import direction_grid, pixel_to_spherical_campvg, spherical_to_direction


def relative_pose(
    r_i: Tensor,
    t_i: Tensor,
    r_j: Tensor,
    t_j: Tensor,
) -> tuple[Tensor, Tensor]:
    """Eq. (6): R_{i→j}, t_{i→j}."""
    r_rel = r_j @ r_i.transpose(-1, -2)
    t_rel = t_j - r_rel @ t_i
    return r_rel, t_rel


def epipolar_v_curve(u: Tensor, width: int, height: int, a_p: float, b_p: float) -> Tensor:
    """Eq. (8): v(u) for epipolar line in target view j."""
    arg = a_p * torch.sin(2 * math.pi * u / width) + torch.cos(2 * math.pi * u / width) / b_p.clamp(min=1e-6)
    return -height / math.pi * torch.atan(arg)


def sample_epipolar_line(
    a_p: float,
    b_p: float,
    width: int,
    height: int,
    k: int,
    device: torch.device,
) -> Tensor:
    """K samples along epipolar curve → [K, 3] directions."""
    u = torch.linspace(0, width - 1, k, device=device)
    v = epipolar_v_curve(u, width, height, torch.tensor(a_p, device=device), torch.tensor(b_p, device=device))
    v = v.clamp(0, height - 1)
    phi, theta = pixel_to_spherical_campvg(u, v, height, width)
    return spherical_to_direction(phi, theta)


def min_distance_to_epipolar_stub(
    p_dir: Tensor,
    samples: Tensor,
) -> Tensor:
    """Eq. (9): min ||p - c_k||_2 over K samples."""
    return torch.cdist(p_dir.unsqueeze(0), samples.unsqueeze(0)).min()


def spherical_epipolar_mask_stub(
    height: int,
    width: int,
    r_i: Tensor,
    t_i: Tensor,
    r_j: Tensor,
    t_j: Tensor,
    *,
    k: int = 250,
    threshold_ratio: float = 0.5,
) -> Tensor:
    """
    Binary mask [H, W] for valid references in view j given source pixel grid.

    Simplified plane coefficients for smoke; full Eq. (7) uses o and p projections.
    """
    device = r_i.device
    r_rel, t_rel = relative_pose(r_i, t_i, r_j, t_j)
    grid = direction_grid(height, width, device=device)
    # stub: use small |cross(p, t_rel)| as plane proximity
    plane_score = torch.linalg.norm(
        torch.cross(grid, t_rel.view(1, 1, 3).expand_as(grid), dim=-1),
        dim=-1,
    )
    diag = math.sqrt(height**2 + width**2) * threshold_ratio
    return (plane_score < diag).float()
