"""Saliency for latent-interpolated fine-grained geometry (paper supplementary Sec. 7, Eq. 14–19).

RelitLRM / DiffusionRenderer are **not** required here: this module only needs multi-view grayscale
(or luminance) maps. When no foreground mask is supplied, edge-based attenuation approximates the
boundary buffer from the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _box_blur(x: Tensor, kernel: int) -> Tensor:
    """Channel-wise average blur; ``x`` is (B, 1, H, W)."""
    if kernel <= 1:
        return x
    k = kernel | 1  # odd
    pad = k // 2
    return F.avg_pool2d(x, k, stride=1, padding=pad)


def raw_texture_saliency(gray: Tensor, box_kernel: int) -> Tensor:
    """Eq. 14–16: local std via box-filtered mean and second moment."""
    mu = _box_blur(gray, box_kernel)
    mu_sq = _box_blur(gray * gray, box_kernel)
    var = (mu_sq - mu * mu).clamp(min=0.0)
    return var.sqrt()


def edge_magnitude(gray: Tensor) -> Tensor:
    """Sobel magnitude in ``[-1,1]``-ish float space."""
    b, _, h, w = gray.shape
    gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ix = F.conv2d(gray, gx, padding=1)
    iy = F.conv2d(gray, gy, padding=1)
    return torch.sqrt(ix * ix + iy * iy + 1e-8)


def gaussian_blur2d(x: Tensor, sigma: float, kernel: int | None = None) -> Tensor:
    """Separable Gaussian blur on (B,1,H,W)."""
    if sigma <= 0:
        return x
    if kernel is None:
        kernel = int(2 * math.ceil(3 * sigma) + 1) | 1
    half = kernel // 2
    t = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g1d = torch.exp(-(t * t) / (2 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g_h = g1d.view(1, 1, 1, kernel)
    g_v = g1d.view(1, 1, kernel, 1)
    pad_h = (kernel // 2, kernel // 2, 0, 0)
    pad_v = (0, 0, kernel // 2, kernel // 2)
    y = F.pad(x, pad_h, mode="reflect")
    y = F.conv2d(y, g_h.expand(x.size(1), -1, -1, -1), groups=x.size(1))
    y = F.pad(y, pad_v, mode="reflect")
    y = F.conv2d(y, g_v.expand(x.size(1), -1, -1, -1), groups=x.size(1))
    return y


def saliency_map(
    gray: Tensor,
    *,
    box_kernel: int = 5,
    gaussian_sigma: float = 2.0,
    edge_attenuation: float = 6.0,
    foreground_mask: Tensor | None = None,
) -> Tensor:
    """Return ``S_final`` (B,1,H,W), nonnegative.

    If ``foreground_mask`` (B,1,H,W) in {0,1} is given, distance-to-boundary weighting (Eq. 17)
    is approximated with distance transform on CPU via ``cv2`` when available; otherwise falls
    back to edge attenuation only.
    """
    s_raw = raw_texture_saliency(gray, box_kernel)
    if foreground_mask is not None:
        w_linear = _boundary_weight_from_mask(foreground_mask)
    else:
        e = edge_magnitude(gray)
        e = e / (e.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6))
        w_linear = torch.exp(-edge_attenuation * e).clamp(0.0, 1.0)
    w = gaussian_blur2d(w_linear, gaussian_sigma)
    return (s_raw * w).clamp(min=0.0)


def _boundary_weight_from_mask(mask: Tensor) -> Tensor:
    """Eq. 17–18 proxy: high weight inside, low near mask boundary."""
    try:
        import cv2  # type: ignore[import-not-found]

        out = []
        m = (mask > 0.5).float()
        for bi in range(m.shape[0]):
            fg = m[bi, 0].detach().cpu().numpy().astype("uint8")
            # distance to nearest zero pixel (background)
            inv = 1 - fg
            if inv.max() < 1:
                out.append(torch.ones_like(m[bi : bi + 1]))
                continue
            dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=3)
            d = torch.from_numpy(dist).to(device=mask.device, dtype=mask.dtype).unsqueeze(0).unsqueeze(0)
            d = d / (d.amax() + 1e-6)
            out.append(d.clamp(0, 1))
        return torch.cat(out, dim=0)
    except Exception:
        e = edge_magnitude(mask.float())
        e = e / (e.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6))
        return (1.0 - e).clamp(0.0, 1.0)


def patch_grid_shape(h: int, w: int, patch: int) -> tuple[int, int]:
    ph, pw = h // patch, w // patch
    return ph, pw


def saliency_patch_scores(s_map: Tensor, patch: int) -> Tensor:
    """Average saliency per patch → (B, P_h, P_w)."""
    b, _, h, w = s_map.shape
    ph, pw = patch_grid_shape(h, w, patch)
    x = s_map[:, :, : ph * patch, : pw * patch]
    x = x.reshape(b, 1, ph, patch, pw, patch).mean(dim=(3, 5))
    return x[:, 0]


def topk_patch_flat_indices(scores: Tensor, top_frac: float) -> Tensor:
    """Flat indices into P_h*P_w grid, shape (B, K)."""
    b, ph, pw = scores.shape
    flat = scores.reshape(b, -1)
    n = flat.shape[1]
    k = max(1, int(math.ceil(top_frac * n)))
    k = min(k, n)
    _, idx = torch.topk(flat, k, dim=-1)
    return idx
