"""Target fitting losses (Eq. 1–4)."""

from __future__ import annotations

import torch
import torch.nn.functional as f
from torch import Tensor


def gaussian_blur_nchw(img: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
    """Separable Gaussian blur for (B, C, H, W)."""
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    g_h = g.view(1, 1, 1, -1).expand(img.shape[1], 1, 1, -1)
    g_v = g.view(1, 1, -1, 1).expand(img.shape[1], 1, -1, 1)
    x = img
    x = f.conv2d(x, g_h, padding=(0, radius), groups=img.shape[1])
    x = f.conv2d(x, g_v, padding=(radius, 0), groups=img.shape[1])
    return x


def blurred_mse_loss(
    rendered: Tensor,
    target: Tensor,
    *,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Tensor:
    """LMSE: mean over frames of ||G(R) - G(I)||^2 (Eq. 1)."""
    if rendered.shape != target.shape:
        raise ValueError("rendered and target must match shape")
    gr = gaussian_blur_nchw(rendered, kernel_size=kernel_size, sigma=sigma)
    gt = gaussian_blur_nchw(target, kernel_size=kernel_size, sigma=sigma)
    return f.mse_loss(gr, gt)


def spatial_offset_regularizer(
    deltas: Tensor,
    adjacency: list[tuple[int, int]],
    canonical_xy: Tensor,
    *,
    sigma: float,
) -> Tensor:
    """Exponential spatial regularization on adjacent control-point offsets (Eq. 2)."""
    if deltas.ndim != 2 or deltas.shape[-1] != 2:
        raise ValueError("deltas must be (N, 2)")
    if not adjacency:
        return deltas.new_zeros(())
    terms: list[Tensor] = []
    for i, j in adjacency:
        w = torch.exp(-((canonical_xy[i] - canonical_xy[j]).pow(2).sum()) / (sigma**2))
        terms.append(w * (deltas[i] - deltas[j]).pow(2).sum())
    return torch.stack(terms).mean()


def g1_joint_penalty(
    prev_handle: Tensor,
    anchor: Tensor,
    next_handle: Tensor,
    *,
    enforce: bool = True,
) -> Tensor:
    """1 - cos(tangent_left, tangent_right) at cubic Bézier joints."""
    if not enforce:
        return anchor.new_zeros(())
    v1 = anchor - prev_handle
    v2 = next_handle - anchor
    n1 = v1.norm().clamp(min=1e-8)
    n2 = v2.norm().clamp(min=1e-8)
    cos = (v1 * v2).sum() / (n1 * n2)
    return 1.0 - cos.clamp(-1.0, 1.0)


def foreground_containment_penalty(
    points_xy: Tensor,
    foreground_mask: Tensor,
    *,
    margin_px: float = 1.0,
) -> Tensor:
    """SDF-style guardrail: penalize control points outside dilated foreground."""
    if foreground_mask.ndim != 2:
        raise ValueError("foreground_mask must be (H, W)")
    h, w = foreground_mask.shape
    # normalize to [-1, 1] for grid_sample
    grid = points_xy.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    grid = grid.view(1, -1, 1, 2)
    mask = foreground_mask.unsqueeze(0).unsqueeze(0)
    sampled = f.grid_sample(mask, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    inside = sampled.view(-1)
    return (margin_px * (1.0 - inside).relu()).pow(2).mean()


def livesvg_total_loss(
    rendered: Tensor,
    target: Tensor,
    *,
    deltas: Tensor,
    adjacency: list[tuple[int, int]],
    canonical_xy: Tensor,
    foreground_mask: Tensor,
    control_points: Tensor,
    g1_triplets: list[tuple[int, int, int]],
    cfg_weights: tuple[float, float, float, float],
    spatial_sigma: float,
    g1_enforce: list[bool],
    sdf_margin: float,
    blur_kernel: int,
    blur_sigma: float,
) -> dict[str, Tensor]:
    """L = λ_mse LMSE + λ_spatial Lspatial + λ_g1 LG1 + λ_sdf LSDF (Eq. 4)."""
    lam_mse, lam_sp, lam_g1, lam_sdf = cfg_weights
    l_mse = blurred_mse_loss(rendered, target, kernel_size=blur_kernel, sigma=blur_sigma)
    l_sp = spatial_offset_regularizer(deltas, adjacency, canonical_xy, sigma=spatial_sigma)
    g1_terms = [
        g1_joint_penalty(control_points[a], control_points[b], control_points[c], enforce=enf)
        for (a, b, c), enf in zip(g1_triplets, g1_enforce, strict=True)
    ]
    l_g1 = torch.stack(g1_terms).mean() if g1_terms else rendered.new_zeros(())
    l_sdf = foreground_containment_penalty(control_points, foreground_mask, margin_px=sdf_margin)
    total = lam_mse * l_mse + lam_sp * l_sp + lam_g1 * l_g1 + lam_sdf * l_sdf
    return {"total": total, "mse": l_mse, "spatial": l_sp, "g1": l_g1, "sdf": l_sdf}
