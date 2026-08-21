"""ErpGS loss terms (Eq. 12–18)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.erpgs.config import ErpGSConfig


def ssim_stub(a: Tensor, b: Tensor) -> Tensor:
    ca = a - a.mean()
    cb = b - b.mean()
    return (2 * (ca * cb).mean() + 1e-3) / (ca.pow(2).mean() + cb.pow(2).mean() + 1e-3)


def color_reconstruction_error(
    pred: Tensor,
    gt: Tensor,
    *,
    lambda_ssim: float,
) -> Tensor:
    """CRE(p) per pixel (Eq. 17)."""
    l1 = (pred - gt).abs()
    ssim_term = 1.0 - ssim_stub(pred, gt)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim_term


def weighted_color_loss(
    pred: Tensor,
    gt: Tensor,
    weight: Tensor,
    mask: Tensor | None,
    cfg: ErpGSConfig,
) -> Tensor:
    """L_color (Eq. 15)."""
    cre = color_reconstruction_error(pred, gt, lambda_ssim=cfg.lambda_ssim)
    w = weight
    if mask is not None:
        w = w * mask
    num = (w * cre).sum()
    den = w.sum().clamp_min(1e-6)
    return num / den


def depth_normal_error(
    n_render: Tensor,
    n_depth: Tensor,
    rgb: Tensor,
    weight: Tensor,
    mask: Tensor | None,
) -> Tensor:
    """DNE(p) with color gradient gating (Eq. 12)."""
    if rgb.dim() == 4:
        gray = rgb.mean(dim=1, keepdim=True)
    else:
        gray = rgb
    gx = gray[..., :, 1:] - gray[..., :, :-1]
    gy = gray[..., 1:, :] - gray[..., :-1, :]
    grad = torch.zeros_like(gray)
    grad[..., :, :-1] += gx.abs()
    grad[..., :, 1:] += gx.abs()
    grad[..., :-1, :] += gy.abs()
    grad[..., 1:, :] += gy.abs()

    dne = grad * (n_render - n_depth).norm(dim=1, keepdim=True)
    w = weight
    if mask is not None:
        w = w * mask
    return (w * dne.squeeze(1)).sum() / w.sum().clamp_min(1e-6)


def scale_regularization(scales: Tensor) -> Tensor:
    """L_s (Eq. 13)."""
    return (scales.pow(2).sum(dim=-1)).mean() / 2.0


def flatten_regularization(scales: Tensor) -> Tensor:
    """L_f — penalize min axis scale (Eq. 13)."""
    smin, _ = scales.min(dim=-1)
    return smin.abs().mean()


def total_loss(
    pred_rgb: Tensor,
    gt_rgb: Tensor,
    n_render: Tensor,
    n_depth: Tensor,
    scales: Tensor,
    weight: Tensor,
    mask: Tensor | None,
    cfg: ErpGSConfig,
    *,
    iteration: int,
) -> dict[str, Tensor]:
    lc = weighted_color_loss(pred_rgb, gt_rgb, weight, mask, cfg)
    ls = scale_regularization(scales)
    ldn = torch.tensor(0.0, device=pred_rgb.device)
    lf = torch.tensor(0.0, device=pred_rgb.device)
    if iteration >= cfg.reg_start_iter:
        ldn = depth_normal_error(n_render, n_depth, gt_rgb, weight, mask)
        lf = flatten_regularization(scales)
    total = lc + cfg.lambda_dn * ldn + cfg.lambda_f * lf + 0.5 * cfg.lambda_s * ls
    return {"L": total, "L_color": lc, "L_dn": ldn, "L_f": lf, "L_s": ls}
