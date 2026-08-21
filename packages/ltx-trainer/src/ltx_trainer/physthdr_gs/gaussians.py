"""HDR 3D Gaussians: geometry + intrinsic reflectance + ambient illumination."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.p2gs.gaussians import _quat_to_rot


class HDRGaussianField(nn.Module):
    """
    Learnable HDR Gaussian primitives with disentangled Hr and La (Eq. 8–11).

    Full 3DGS geometry (μ, Σ, α) plus per-point reflectance and illumination.
    """

    def __init__(self, num_points: int, *, init_radius: float = 1.5) -> None:
        super().__init__()
        n = num_points
        pts = torch.randn(n, 3) * 0.4
        pts[:, 2] = pts[:, 2].abs() + 1.0
        self._xyz = nn.Parameter(pts * init_radius / max(init_radius, 1e-3))
        self._hr = nn.Parameter(torch.rand(n, 3) * 0.5 + 0.25)
        self._la = nn.Parameter(torch.rand(n, 3) * 0.5 + 0.5)
        self._scaling = nn.Parameter(torch.full((n, 3), math.log(0.05)))
        rot = torch.zeros(n, 4)
        rot[:, 0] = 1.0
        self._rotation = nn.Parameter(rot)
        self._opacity = nn.Parameter(torch.full((n, 1), math.log(0.3 / 0.7)))
        self._max_points = max(n * 4, 50_000)

    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]

    def get_xyz(self) -> Tensor:
        return self._xyz

    def get_hr(self) -> Tensor:
        return self._hr.clamp(min=0.0)

    def get_la(self) -> Tensor:
        return self._la.clamp(min=0.0)

    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self._opacity)

    def get_scales(self) -> Tensor:
        return torch.exp(self._scaling).clamp(min=1e-4, max=1.0)

    def get_rotation(self) -> Tensor:
        return self._rotation

    @classmethod
    def from_point_cloud(
        cls,
        points: Tensor,
        *,
        colors_hr: Tensor | None = None,
        scale: float = 0.06,
        max_points: int = 50_000,
    ) -> HDRGaussianField:
        n = points.shape[0]
        if n > max_points:
            idx = torch.randperm(n, device=points.device)[:max_points]
            points = points[idx]
            n = points.shape[0]
        g = cls(n, init_radius=1.0)
        with torch.no_grad():
            g._xyz.copy_(points)
            if colors_hr is not None:
                hr = colors_hr[:n] if colors_hr.shape[0] >= n else colors_hr
                g._hr.copy_(hr.clamp(min=0.01))
            g._scaling.fill_(math.log(scale))
        g._max_points = max_points
        return g

    def _register(self, name: str, tensor: Tensor) -> None:
        self.register_parameter(name, nn.Parameter(tensor))

    def clone_points(self, mask: Tensor) -> None:
        """Duplicate Gaussians selected by boolean mask (densification)."""
        if not mask.any():
            return
        n_sel = int(mask.sum().item())
        if self.num_points + n_sel > self._max_points:
            budget = self._max_points - self.num_points
            if budget <= 0:
                return
            scores = mask.nonzero(as_tuple=False).squeeze(1)
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[scores[:budget]] = True

        for name in ("_xyz", "_hr", "_la", "_scaling", "_rotation", "_opacity"):
            p = getattr(self, name)
            self._register(name, torch.cat([p.data, p.data[mask]], dim=0))

    def prune_points(self, mask: Tensor) -> None:
        """Remove Gaussians where mask is True."""
        keep = ~mask
        if keep.all():
            return
        for name in ("_xyz", "_hr", "_la", "_scaling", "_rotation", "_opacity"):
            p = getattr(self, name)
            self._register(name, p.data[keep])

    def covariance_3d(self) -> Tensor:
        s = self.get_scales()
        R = _quat_to_rot(self._rotation)
        S = torch.diag_embed(s)
        return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
