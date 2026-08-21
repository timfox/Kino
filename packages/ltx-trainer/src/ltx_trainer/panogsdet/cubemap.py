"""Cube-map semantic rendering for L_sem (Eq. 6)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.panogsdet.gaussian_lift import SemanticGaussianState

FACE_NAMES = ("up", "down", "front", "back", "left", "right")


def _face_basis(face: str) -> tuple[Tensor, Tensor, Tensor]:
    """Return (u_axis, v_axis, normal) unit vectors on CPU; caller moves to device."""
    axes = {
        "front": (
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ),
        "back": (
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, -1.0]),
        ),
        "left": (
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([-1.0, 0.0, 0.0]),
        ),
        "right": (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ),
        "up": (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ),
        "down": (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0.0, -1.0, 0.0]),
        ),
    }
    return axes[face]


def render_semantic_cubemap(
    state: SemanticGaussianState,
    face_size: int = 64,
    *,
    max_points: int = 2048,
) -> dict[str, Tensor]:
    """
    Splat Gaussian class logits onto six cube faces → [B,C,H,W] per face.
    """
    device = state.centers.device
    b = state.centers.shape[0]
    n = min(state.num_gaussians, max_points)
    cls = state.category_logits[:, :n]
    opacity = state.opacity[:, :n].unsqueeze(-1)
    centers = state.centers[:, :n]
    out: dict[str, Tensor] = {}
    grid = torch.linspace(-1, 1, face_size, device=device)
    gv, gu = torch.meshgrid(grid, grid, indexing="ij")
    for face in FACE_NAMES:
        u_ax, v_ax, normal = _face_basis(face)
        u_ax = u_ax.to(device)
        v_ax = v_ax.to(device)
        normal = normal.to(device)
        plane_pts = normal + gu.unsqueeze(-1) * u_ax + gv.unsqueeze(-1) * v_ax  # [H,W,3]
        maps = []
        for bi in range(b):
            # plane_pts [H,W,3], centers [n,3]
            diff = plane_pts.unsqueeze(2) - centers[bi].view(1, 1, n, 3)
            dist = diff.pow(2).sum(dim=-1)
            w = torch.exp(-dist * 8.0) * opacity[bi].view(1, 1, n)
            w = w / w.sum(dim=2, keepdim=True).clamp(min=1e-6)
            feat_map = (w.unsqueeze(-1) * cls[bi].view(1, 1, n, -1)).sum(dim=2)
            maps.append(feat_map.permute(2, 0, 1))
        out[face] = torch.stack(maps, dim=0)
    return out


def erp_semantic_to_cubemap(sem_erp: Tensor, face_size: int = 64) -> dict[str, Tensor]:
    """Project ERP semantic logits to cube faces (GT proxy for training smoke)."""
    b, c, h, w = sem_erp.shape
    # lightweight: downsample ERP per face via fixed crops (stub)
    faces = {}
    for name in FACE_NAMES:
        faces[name] = torch.nn.functional.adaptive_avg_pool2d(sem_erp, (face_size, face_size))
    return faces
