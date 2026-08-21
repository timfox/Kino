"""Official RAIM Track 2 leaderboard score (Eq. 1, arXiv:2604.09030)."""

from __future__ import annotations


def leaderboard_score(psnr: float, ssim: float, lpips: float) -> float:
    """
    Combine PSNR (dB), SSIM [0,1], and LPIPS [0,~0.4] into challenge score.

    Paper Eq. (1) is implemented as:
      30·PSNR/50 + 22.5·(SSIM−0.5)/0.5 + 30·(1−LPIPS/0.4)
    which matches published WHU-VIP Stage-1 anchor (~58.249).
    """
    term_psnr = 30.0 * psnr / 50.0
    term_ssim = 22.5 * (ssim - 0.5) / 0.5
    term_lpips = 30.0 * (1.0 - lpips / 0.4)
    return float(term_psnr + term_ssim + term_lpips)


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    import math

    if mse <= 0.0:
        return 99.0
    return float(10.0 * math.log10((peak * peak) / mse))


def ssim_proxy(pred, ref) -> float:
    """Lightweight SSIM proxy when pytorch-msssim is unavailable."""
    try:
        import torch
        import torch.nn.functional as F

        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        else:
            pred = pred.detach()
        if not isinstance(ref, torch.Tensor):
            ref = torch.as_tensor(ref)
        else:
            ref = ref.detach()
        c1, c2 = 0.01**2, 0.03**2
        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(ref, 3, 1, 1)
        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(ref * ref, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * ref, 3, 1, 1) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        return float((num / den).mean().clamp(0, 1))
    except Exception:
        import numpy as np

        p = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        r = ref.detach().cpu().numpy() if hasattr(ref, "detach") else np.asarray(ref)
        corr = np.corrcoef(p.flatten(), r.flatten())[0, 1]
        return float(max(0.0, min(1.0, (corr + 1) * 0.5)))


def lpips_proxy(pred, ref) -> float:
    """MSE-scaled LPIPS stand-in for smoke tests."""
    import torch

    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred, dtype=torch.float32)
    if not isinstance(ref, torch.Tensor):
        ref = torch.as_tensor(ref, dtype=torch.float32)
    mse = float((pred - ref).pow(2).mean())
    return float(min(0.4, mse**0.5 * 2.0))
