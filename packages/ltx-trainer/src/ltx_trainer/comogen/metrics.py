"""Evaluation metrics for CoMoGen (Sec. 5, Tables 1–3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def psnr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    mse = ((pred - target) ** 2).mean()
    if mse <= 0:
        return float("inf")
    return float(10.0 * torch.log10(1.0 / (mse + eps)).item())


def ssim_proxy(pred: Tensor, target: Tensor) -> float:
    mu_p, mu_t = pred.mean(), target.mean()
    var_p = ((pred - mu_p) ** 2).mean()
    var_t = ((target - mu_t) ** 2).mean()
    cov = ((pred - mu_p) * (target - mu_t)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(
        ((2 * mu_p * mu_t + c1) * (2 * cov + c2) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2) + 1e-8))
        .clamp(0, 1)
        .item()
    )


def lpips_proxy(pred: Tensor, target: Tensor) -> float:
    return float(torch.abs(pred - target).mean().item())


def fvd_proxy(pred: Tensor, target: Tensor) -> float:
    if pred.dim() == 5:
        pred, target = pred[0], target[0]
    fp = F.adaptive_avg_pool2d(pred, (4, 4)).flatten(1)
    ft = F.adaptive_avg_pool2d(target, (4, 4)).flatten(1)
    return float(((fp - ft).pow(2).mean(dim=1).mean() * 1000.0).item())


def tracking_jf_hota(pred_masks: Tensor, gt_masks: Tensor) -> dict[str, float]:
    """J, F, HOTA proxies from mask IoU over time."""
    pred = pred_masks > 0.5
    gt = gt_masks > 0.5
    inter = (pred & gt).float().sum(dim=(-2, -1))
    union = (pred | gt).float().sum(dim=(-2, -1)).clamp(min=1)
    j = float((inter / union).mean().item() * 100)
    # contour proxy via edge dilate
    f = float((inter / (gt.float().sum(dim=(-2, -1)).clamp(min=1))).mean().item() * 100)
    hota = float((j * f / 100.0) ** 0.5)
    return {"j": j, "f": f, "j_and_f": (j + f) / 2, "hota": hota}


def evaluate_video(
    pred: Tensor,
    target: Tensor,
    *,
    pred_masks: Tensor | None = None,
    gt_masks: Tensor | None = None,
) -> dict[str, float]:
    out = {
        "ssim": ssim_proxy(pred, target),
        "psnr": psnr(pred, target),
        "lpips": lpips_proxy(pred, target),
        "fvd": fvd_proxy(pred, target),
    }
    if pred_masks is not None and gt_masks is not None:
        out.update(tracking_jf_hota(pred_masks, gt_masks))
    return out
