"""Anisotropic 3D Gaussian parameters (Sec. 3.2, 4.1)."""

from __future__ import annotations

import torch
from torch import Tensor


def build_covariance(
    log_scale: Tensor,
    quat: Tensor | None = None,
) -> Tensor:
    """Σ = R S S^T R^T with diagonal S from exp(log_scale)."""
    scale = torch.exp(log_scale)
    s = torch.diag_embed(scale)
    if quat is None:
        r = torch.eye(3, device=log_scale.device, dtype=log_scale.dtype)
        if log_scale.dim() > 1:
            r = r.unsqueeze(0).expand(log_scale.shape[0], -1, -1)
    else:
        r = quaternion_to_rotation(quat)
    return r @ s @ s.transpose(-1, -2) @ r.transpose(-1, -2)


def quaternion_to_rotation(q: Tensor) -> Tensor:
    """Unit quaternion (w,x,y,z) → R ∈ SO(3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def project_covariance_stub(
    sigma: Tensor,
    jacobian: Tensor | None = None,
) -> Tensor:
    """Σ' = J V Σ V^T J^T (2×2 footprint stub)."""
    if jacobian is None:
        jacobian = torch.eye(3, device=sigma.device, dtype=sigma.dtype)[:2]
    if sigma.dim() == 2:
        return jacobian @ sigma @ jacobian.t()
    return jacobian @ sigma @ jacobian.transpose(-1, -2)
