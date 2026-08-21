"""Structural edges, misalignment masks, and latent weight maps (Sec. 3.2, Eq. 4–5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _as_hw(tensor: Tensor) -> Tensor:
    """Collapse leading batch dims to a single (H, W) map."""
    out = tensor
    while out.dim() > 2:
        out = out.squeeze(0)
    return out


def _dilate_mask(mask: Tensor, kernel: int) -> Tensor:
    m = _as_hw(mask).unsqueeze(0).unsqueeze(0)
    m = F.max_pool2d(m, kernel, stride=1, padding=kernel // 2)
    return m.squeeze(0).squeeze(0)


def log_luminance_highpass(image: Tensor, *, kernel_size: int = 9, sigma: float = 0.15) -> Tensor:
    """H(I): log-luminance minus smoothed version (illumination-invariant structure)."""
    from ltx_trainer.controllight.retinex import luminance_y, smooth_illumination

    y = luminance_y(image.clamp(0, 1))
    log_y = torch.log(y.clamp(min=1e-4))
    smooth = smooth_illumination(log_y, kernel_size=kernel_size, sigma=sigma)
    return log_y - smooth


def structural_edge_response(image: Tensor) -> Tensor:
    """E(I) = ||∇H(I)||_1 (Sec. 3.2). Returns (H, W) or (B, H, W)."""
    h = log_luminance_highpass(image)
    if h.dim() == 2:
        h = h.unsqueeze(0).unsqueeze(0)
    elif h.dim() == 3:
        h = h.unsqueeze(1)
    gx = h[..., :, 1:] - h[..., :, :-1]
    gy = h[..., 1:, :] - h[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    edges = (gx.abs() + gy.abs()).squeeze(1)
    while edges.dim() > 3:
        edges = edges.squeeze(0)
    return edges


def edge_difference_map(a: Tensor, b: Tensor) -> Tensor:
    """I_edge-diff(A,B) = |E(A) - E(B)|."""
    return (structural_edge_response(a) - structural_edge_response(b)).abs()


def binary_edge_map(edge_response: Tensor, quantile: float = 0.85) -> Tensor:
    """Binarize edge response at per-image quantile."""
    e = _as_hw(edge_response)
    thr = torch.quantile(e.flatten(), quantile)
    return (e >= thr).float()


def distance_to_input_edges(input_edges: Tensor) -> Tensor:
    """Distance transform D0 to nearest edge in B0 (approximate via iterative dilation)."""
    b = (input_edges > 0.5).float()
    b = _as_hw(b).unsqueeze(0)
    inv = 1.0 - b
    dist = torch.zeros_like(b)
    for _ in range(32):
        dil = F.max_pool2d(inv, 3, stride=1, padding=1)
        dist = torch.where(inv > 0.5, dist + 1.0, dist)
        inv = torch.where(dil > 0.5, torch.zeros_like(inv), inv)
    return dist


def unreliable_target_mask(
    b0: Tensor,
    bs: Tensor,
    *,
    dist_threshold: float = 3.0,
) -> Tensor:
    """Eq. (4): M_s(p) = 1[B_s=1 and D0(p) > d]."""
    d0 = distance_to_input_edges(b0)
    return ((bs > 0.5) & (d0 > dist_threshold)).float()


def misalignment_weight_map(
    i0: Tensor,
    is_target: Tensor,
    *,
    dist_threshold: float = 3.0,
    alpha: float = 0.8,
    wmin: float = 0.2,
    dilate: int = 3,
) -> Tensor:
    """Eq. (5): W_s = clip(1 - α M_s, wmin, 1) with optional dilation on M_s."""
    e0 = structural_edge_response(i0)
    es = structural_edge_response(is_target)
    b0 = binary_edge_map(e0)
    bs = binary_edge_map(es)
    ms = unreliable_target_mask(b0, bs, dist_threshold=dist_threshold)
    if dilate > 1:
        ms = _dilate_mask(ms, dilate)
    ms = _as_hw(ms)
    return (1.0 - alpha * ms).clamp(wmin, 1.0)


def resize_weight_to_latent(weight: Tensor, latent_shape: tuple[int, ...]) -> Tensor:
    """Resize image-space W_s to latent spatial resolution (fW_s)."""
    if weight.dim() == 2:
        weight = weight.unsqueeze(0).unsqueeze(0)
    elif weight.dim() == 3:
        weight = weight.unsqueeze(1)
    h, w = latent_shape[-2], latent_shape[-1]
    return F.interpolate(weight, size=(h, w), mode="bilinear", align_corners=False)
