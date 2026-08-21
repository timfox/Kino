"""WS-PSNR weighting proxy (Sun et al., IEEE SPL 2017; JVET evaluation)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.nvc_erp_qpa.config import WS_PSNR_YUV_WEIGHTS


def spherical_pixel_weights(height: int, width: int, *, device: torch.device | None = None) -> Tensor:
    """Per-pixel weight ∝ cos φ (area element on sphere), shape (H, W)."""
    rows = torch.arange(height, dtype=torch.float32, device=device)
    if height > 1:
        phi = math.pi / 2 - math.pi * rows / (height - 1)
    else:
        phi = torch.zeros(1, device=device)
    w_row = torch.cos(phi).clamp(min=1e-6)
    return w_row.view(-1, 1).expand(height, width)


def ws_mse(y: Tensor, y_hat: Tensor, weights: Tensor | None = None) -> Tensor:
    """Weighted MSE; inputs B×C×H×W in [0,1]."""
    if weights is None:
        weights = spherical_pixel_weights(y.shape[-2], y.shape[-1], device=y.device)
    w = weights.to(y.device).view(1, 1, *weights.shape)
    return (w * (y - y_hat).pow(2)).sum() / w.sum().clamp(min=1e-8)


def ws_psnr_db(y: Tensor, y_hat: Tensor, weights: Tensor | None = None, peak: float = 1.0) -> float:
    mse = float(ws_mse(y, y_hat, weights).item())
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(peak * peak / mse)


def yuv_ws_psnr_proxy(
    y: Tensor,
    y_hat: Tensor,
    *,
    weights: tuple[int, int, int] = WS_PSNR_YUV_WEIGHTS,
) -> float:
    """Single-plane proxy with Y weight 6, chroma 1 (Table V-A)."""
    w_map = spherical_pixel_weights(y.shape[-2], y.shape[-1], device=y.device)
    mse_y = ws_mse(y, y_hat, w_map)
    total_w = sum(weights)
    # stub: chroma planes omitted; scale Y-only WS-PSNR by Wy/total
    mse_eff = mse_y * (weights[0] / total_w)
    mse_v = float(mse_eff.item())
    if mse_v <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse_v)
