"""Background-masked fidelity metrics (Sec. IV-B)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _background_mask(mask: Tensor, target: Tensor) -> Tensor:
    """mask 1=edited; background weight = 1 - mask."""
    m = mask.to(dtype=target.dtype, device=target.device)
    while m.dim() < target.dim():
        m = m.unsqueeze(-1)
    return 1.0 - m


def psnr_masked(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> float:
    w = _background_mask(mask, pred)
    mse = ((pred - target) ** 2 * w).sum() / (w.sum() * pred.shape[-1] + eps)
    if mse <= 0:
        return float("inf")
    return float(10.0 * torch.log10(1.0 / mse).item())


def ssim_proxy_masked(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    """Window-free SSIM proxy on background pixels."""
    w = _background_mask(mask, pred)
    mu_p = (pred * w).sum() / (w.sum() + 1e-8)
    mu_t = (target * w).sum() / (w.sum() + 1e-8)
    var_p = (((pred - mu_p) ** 2) * w).sum() / (w.sum() + 1e-8)
    var_t = (((target - mu_t) ** 2) * w).sum() / (w.sum() + 1e-8)
    cov = (((pred - mu_p) * (target - mu_t)) * w).sum() / (w.sum() + 1e-8)
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    den = (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    return float((num / (den + 1e-8)).clamp(0, 1).item())


def lpips_proxy_masked(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    """L1-based perceptual proxy on background."""
    w = _background_mask(mask, pred)
    return float((torch.abs(pred - target) * w).sum() / (w.sum() * pred.shape[-1] + 1e-8))


def vfid_proxy(pred: Tensor, target: Tensor) -> float:
    """Lightweight video FID proxy from per-frame feature deltas."""
    # [T, C, H, W] or [B, T, C, H, W]
    if pred.dim() == 5:
        pred = pred[0]
        target = target[0]
    feats_p = F.adaptive_avg_pool2d(pred, (4, 4)).flatten(1)
    feats_t = F.adaptive_avg_pool2d(target, (4, 4)).flatten(1)
    diff = (feats_p - feats_t).pow(2).mean(dim=1)
    return float((diff.mean() * 1000.0).item())


def clip_i_proxy(pred: Tensor, target: Tensor) -> float:
    """Frame-to-frame consistency proxy."""
    if pred.dim() >= 4 and pred.shape[0] > 1:
        a = pred[:-1].flatten(1)
        b = pred[1:].flatten(1)
        cos = F.cosine_similarity(a, b, dim=1).mean()
        return float(cos.clamp(0, 1).item())
    return 1.0


def clip_t_proxy(prompt: str) -> float:
    """Prompt alignment stub (length-normalized)."""
    return float(min(0.35, 0.2 + len(prompt.split()) * 0.02))


def evaluate_background_metrics(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    prompt: str = "",
) -> dict[str, float]:
    return {
        "psnr": psnr_masked(pred, target, mask),
        "ssim": ssim_proxy_masked(pred, target, mask),
        "lpips": lpips_proxy_masked(pred, target, mask),
        "vfid": vfid_proxy(pred, target),
        "clip_i": clip_i_proxy(pred, target),
        "clip_t": clip_t_proxy(prompt),
    }
