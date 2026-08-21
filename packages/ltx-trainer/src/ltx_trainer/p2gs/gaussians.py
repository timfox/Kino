"""3D Gaussian scene representation with linear HDR radiance (P2GS Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _quat_to_rot(q: Tensor) -> Tensor:
    """Unit quaternion ``[N, 4]`` (w, x, y, z) → rotation ``[N, 3, 3]``."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


class GaussianModel(nn.Module):
    """Explicit 3D Gaussians with linear HDR DC color (SH degree 0)."""

    def __init__(
        self,
        xyz: Tensor,
        features: Tensor,
        *,
        scales: Tensor | None = None,
        rotations: Tensor | None = None,
        opacity: Tensor | None = None,
    ) -> None:
        super().__init__()
        n = xyz.shape[0]
        self._xyz = nn.Parameter(xyz.float())
        self._features = nn.Parameter(features.float().clamp(min=0.0))
        if scales is None:
            scales = torch.full((n, 3), -2.3)
        if rotations is None:
            rot = torch.zeros(n, 4)
            rot[:, 0] = 1.0
            rotations = rot
        if opacity is None:
            opacity = torch.full((n, 1), -2.0)
        self._scaling = nn.Parameter(scales.float())
        self._rotation = nn.Parameter(rotations.float())
        self._opacity = nn.Parameter(opacity.float())

    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]

    @property
    def xyz(self) -> Tensor:
        return self._xyz

    @property
    def linear_rgb(self) -> Tensor:
        """Per-Gaussian linear HDR radiance ``[N, 3]``."""
        return self._features.clamp(min=0.0)

    def scales(self) -> Tensor:
        return torch.exp(self._scaling).clamp(min=1e-4, max=1.0)

    def opacity(self) -> Tensor:
        return torch.sigmoid(self._opacity).clamp(1e-4, 1.0 - 1e-4)

    def covariance_3d(self) -> Tensor:
        """Diagonal scale + rotation → ``[N, 3, 3]`` covariance."""
        s = self.scales()
        R = _quat_to_rot(self._rotation)
        S = torch.diag_embed(s)
        return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    @classmethod
    def from_point_cloud(
        cls,
        points: Tensor,
        *,
        colors: Tensor | None = None,
        scale: float = 0.05,
    ) -> GaussianModel:
        n = points.shape[0]
        if colors is None:
            colors = torch.full((n, 3), 0.5)
        log_scale = torch.log(torch.full((n, 3), scale))
        return cls(points, colors, scales=log_scale)

    @classmethod
    def random_init(
        cls,
        num_points: int,
        cameras_center: Tensor,
        *,
        radius: float = 2.0,
        device: torch.device | str = "cpu",
    ) -> GaussianModel:
        """Initialize Gaussians in a ball in front of the camera rig."""
        c = cameras_center.to(device)
        pts = c.unsqueeze(0) + torch.randn(num_points, 3, device=device) * radius
        pts[:, 2] = pts[:, 2].abs() + 3.0
        colors = torch.rand(num_points, 3, device=device) * 0.5 + 0.25
        return cls.from_point_cloud(pts, colors=colors, scale=0.08).to(device)
