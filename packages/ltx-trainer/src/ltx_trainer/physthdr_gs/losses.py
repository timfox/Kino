"""PhysHDR-GS loss functions (Sec. 4.2, 4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.physthdr_gs.rasterize import gaussian_blur


@dataclass
class PhysHDRLossConfig:
    lambda_rec: float = 1.0
    lambda_cons: float = 0.5
    lambda_unit: float = 0.0
    gamma_mse: float = 0.2


def _ssim_simple(pred: Tensor, target: Tensor) -> Tensor:
    mu_p = pred.mean(dim=(-2, -1), keepdim=True)
    mu_t = target.mean(dim=(-2, -1), keepdim=True)
    sig_p = ((pred - mu_p) ** 2).mean(dim=(-2, -1), keepdim=True)
    sig_t = ((target - mu_t) ** 2).mean(dim=(-2, -1), keepdim=True)
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=(-2, -1), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)
    den = (mu_p**2 + mu_t**2 + c1) * (sig_p + sig_t + c2)
    return (num / den).mean()


def reconstruction_loss(
    preds: dict[str, Tensor],
    target: Tensor,
    *,
    gamma: float = 0.2,
) -> Tensor:
    """Lrec over {ILDR, IIG_LDR, IGI_LDR} (Eq. 15)."""
    total = torch.zeros((), device=target.device)
    for key in ("ildr", "iig", "igi"):
        p = preds[key]
        mse = F.mse_loss(p, target)
        dssim = 1.0 - _ssim_simple(p, target)
        total = total + gamma * mse + dssim
    return total / 3.0


def hdr_consistency_loss(ihdr_scaled: Tensor, ihdr_relit: Tensor) -> Tensor:
    """Lcons = ||G(IHDR×t) − G(ˆIHDR)||_1 (Eq. 16)."""
    a = gaussian_blur(ihdr_scaled)
    b = gaussian_blur(ihdr_relit)
    return (a - b).abs().mean()


def unit_exposure_regularization(ihdr: Tensor, exposure: float) -> Tensor:
    """Synthetic-only regularization when λ3 > 0."""
    scaled = ihdr * exposure
    return scaled.var()


class PhysHDRLoss:
    def __init__(self, cfg: PhysHDRLossConfig | None = None) -> None:
        self.cfg = cfg or PhysHDRLossConfig()

    def __call__(
        self,
        *,
        preds: dict[str, Tensor],
        target_ldr: Tensor,
        ihdr_scaled: Tensor,
        ihdr_relit: Tensor,
        ihdr: Tensor,
        exposure: float,
        use_cons: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        lrec = reconstruction_loss(preds, target_ldr, gamma=self.cfg.gamma_mse)
        lcons = hdr_consistency_loss(ihdr_scaled, ihdr_relit) if use_cons else torch.zeros((), device=lrec.device)
        lunit = unit_exposure_regularization(ihdr, exposure) if self.cfg.lambda_unit > 0 else torch.zeros((), device=lrec.device)
        total = self.cfg.lambda_rec * lrec + self.cfg.lambda_cons * lcons + self.cfg.lambda_unit * lunit
        stats = {
            "loss_total": float(total.detach()),
            "loss_rec": float(lrec.detach()),
            "loss_cons": float(lcons.detach()),
            "loss_unit": float(lunit.detach()),
        }
        return total, stats
