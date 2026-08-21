"""Training losses: perceptual proxy + temporal consistency (Eq. 6–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.msfet_e2v.config import MSFETE2VConfig


def reconstruction_loss(pred: Tensor, target: Tensor) -> Tensor:
    """LPIPS proxy — L2 on intensity frames (full training uses VGG-LPIPS)."""
    return F.mse_loss(pred, target)


def temporal_consistency_loss(
    pred_k: Tensor,
    pred_km1: Tensor,
    flow: Tensor,
    *,
    alpha: float = 50.0,
) -> Tensor:
    """Eq. (6) with synthetic flow warp (EVREAL / E2VID-style)."""
    warped = warp_with_flow(pred_km1, flow)
    mask = torch.exp(-alpha * F.mse_loss(pred_k, warped, reduction="none").mean(dim=(1, 2, 3), keepdim=True))
    return (mask * (pred_k - warped).abs()).mean()


def warp_with_flow(img: Tensor, flow: Tensor) -> Tensor:
    """Bilinear sample img using per-pixel flow (B, 2, H, W)."""
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx + flow[:, 0] / max(w - 1, 1), yy + flow[:, 1] / max(h - 1, 1)], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def total_loss(
    preds: Tensor,
    targets: Tensor,
    flows: Tensor | None = None,
    cfg: MSFETE2VConfig | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Eq. (7) over a short clip."""
    cfg = cfg or MSFETE2VConfig()
    lr = reconstruction_loss(preds, targets)
    parts: dict[str, float] = {"reconstruction": float(lr.detach())}
    if flows is None or preds.shape[0] < 2:
        return lr, parts
    lt = temporal_consistency_loss(preds[1:], preds[:-1], flows[1:], alpha=cfg.occlusion_alpha)
    total = lr + cfg.temporal_loss_weight * lt
    parts["temporal"] = float(lt.detach())
    parts["total"] = float(total.detach())
    return total, parts
