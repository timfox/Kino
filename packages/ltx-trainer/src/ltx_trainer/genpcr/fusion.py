"""Geometric-color descriptor fusion (Sec. 3.6, Eq. 12)."""

from __future__ import annotations

import torch
from torch import Tensor


def pca_reduce(feat: Tensor, out_dim: int) -> Tensor:
    """Stub PCA: linear projection to out_dim (per-point)."""
    if feat.shape[-1] == out_dim:
        return feat
    proj = torch.randn(feat.shape[-1], out_dim, device=feat.device, dtype=feat.dtype) * 0.02
    return feat @ proj


def fuse_geo_color(
    f_geo: Tensor,
    f_rgb: Tensor,
    *,
    omega: float = 0.5,
    rgb_dim: int | None = None,
) -> Tensor:
    """
    Eq. (12): f̃ = [ω·f_geo ; (1-ω)·PCA(f_rgb)].

    f_geo, f_rgb: N×D tensors.
    """
    d_geo = f_geo.shape[-1]
    target_rgb = rgb_dim if rgb_dim is not None else d_geo
    f_rgb_c = pca_reduce(f_rgb, target_rgb)
    return torch.cat([omega * f_geo, (1.0 - omega) * f_rgb_c], dim=-1)


def xyz_rgb_point_cloud(
    points: Tensor,
    rgb_image: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    """6D color point cloud stub: XYZ + sampled RGB from generated image."""
    n = points.shape[0]
    u = torch.randint(0, width, (n,), device=points.device)
    v = torch.randint(0, height, (n,), device=points.device)
    img = rgb_image.squeeze(0) if rgb_image.dim() == 4 else rgb_image
    if img.shape[0] == 3:
        colors = img[:, v, u].t()
    else:
        colors = img[v, u].unsqueeze(-1).expand(n, 3)
    return torch.cat([points, colors], dim=-1)
