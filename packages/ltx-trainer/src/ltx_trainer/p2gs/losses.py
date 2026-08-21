"""P2GS unified loss and driving-scene metrics (Eq. 7–10, Appendix C)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.p2gs.photometric import ViewPhotometricParams, srgb_to_linear


@dataclass
class P2GSLossConfig:
    lambda_dssim: float = 0.2
    lambda_exp: float = 0.01
    lambda_escale: float = 0.01
    lambda_evar: float = 0.1
    lambda_gamma: float = 0.1


def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g).view(1, 1, size, size)


def ssim_map(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    """SSIM in ``[0, 1]``; returns scalar mean."""
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    c = img1.shape[1]
    device, dtype = img1.device, img1.dtype
    w = _gaussian_window(window_size, 1.5, device, dtype).expand(c, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1 = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1_sq
    sigma2 = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu12
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return ssim.mean()


class P2GSLosses:
    """``L_total = L_photo + λ_exp L_exp + L_reg`` (Eq. 7)."""

    def __init__(self, cfg: P2GSLossConfig | None = None) -> None:
        self.cfg = cfg or P2GSLossConfig()

    def photometric(self, pred_ldr: Tensor, gt_ldr: Tensor) -> Tensor:
        l1 = F.l1_loss(pred_ldr, gt_ldr)
        dssim = 1.0 - ssim_map(pred_ldr, gt_ldr)
        return (1.0 - self.cfg.lambda_dssim) * l1 + self.cfg.lambda_dssim * dssim

    def relative_exposure(self, hdr_i: Tensor, hdr_j: Tensor, *, alpha_ij: Tensor) -> Tensor:
        """PIR consistency ``‖ α_ij Î_i − Î_j ‖_1`` (Eq. 9)."""
        return F.l1_loss(alpha_ij * hdr_i, hdr_j)

    def regularization(self, view_params: ViewPhotometricParams) -> Tensor:
        l_scale, l_var, l_gamma = view_params.regularization()
        c = self.cfg
        return c.lambda_escale * l_scale + c.lambda_evar * l_var + c.lambda_gamma * l_gamma

    def total(
        self,
        pred_ldr: Tensor,
        gt_ldr: Tensor,
        exp_pairs: list[tuple[Tensor, Tensor, Tensor]],
        view_params: ViewPhotometricParams,
    ) -> tuple[Tensor, dict[str, float]]:
        l_photo = self.photometric(pred_ldr, gt_ldr)
        if exp_pairs:
            l_exp = sum(self.relative_exposure(a, b, alpha_ij=c) for a, b, c in exp_pairs) / len(exp_pairs)
        else:
            l_exp = torch.tensor(0.0, device=pred_ldr.device)
        l_reg = self.regularization(view_params)
        total = l_photo + self.cfg.lambda_exp * l_exp + l_reg
        stats = {
            "l_photo": float(l_photo.detach()),
            "l_exp": float(l_exp.detach()),
            "l_reg": float(l_reg.detach()),
            "l_total": float(total.detach()),
        }
        return total, stats


def build_hdr_pairs(
    hdr_by_view: dict[int, Tensor],
    view_params: ViewPhotometricParams,
    pairs: list[tuple[int, int]],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Build ``(Î_i, Î_j, α_ij)`` with ``α_ij = e_j / e_i`` (Eq. 2, 9)."""
    out: list[tuple[Tensor, Tensor, Tensor]] = []
    e = view_params.exposure.detach()
    for i, j in pairs:
        alpha = (e[j] / e[i].clamp(min=1e-6)).detach()
        out.append((hdr_by_view[i], hdr_by_view[j], alpha))
    return out


def _luminance_bt601(rgb: Tensor) -> Tensor:
    """``[3, H, W]`` or ``[T, 3, H, W]`` → luminance."""
    if rgb.ndim == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]


def hdr_inconsistency_score(
    ldr_sequence: Tensor,
    exposure_scales: Tensor,
) -> float:
    """
    HDR Inconsistency Score (Appendix C.1, Eq. 16).

    Args:
        ldr_sequence: ``[T, 3, H, W]`` rendered LDR frames.
        exposure_scales: ``[T]`` per-frame exposure.
    """
    if ldr_sequence.shape[0] < 2:
        return 0.0
    total = 0.0
    for t in range(ldr_sequence.shape[0] - 1):
        r0 = srgb_to_linear(ldr_sequence[t])
        r1 = srgb_to_linear(ldr_sequence[t + 1])
        lin0 = r0 * exposure_scales[t]
        lin1 = r1 * exposure_scales[t + 1]
        total += float((lin0 - lin1).pow(2).mean().sqrt())
    return total / (ldr_sequence.shape[0] - 1)


def std_luminance(ldr_views: Tensor) -> float:
    """
    Std-Luminance across views (Appendix C.2, Eq. 17–19).

    Args:
        ldr_views: ``[N, 3, H, W]`` or list stacked.
    """
    if ldr_views.ndim == 3:
        ldr_views = ldr_views.unsqueeze(0)
    means = [_luminance_bt601(ldr_views[i]).mean().item() for i in range(ldr_views.shape[0])]
    if len(means) < 2:
        return 0.0
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / len(means)
    return float(var**0.5)
