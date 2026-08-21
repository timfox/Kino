"""Stage-2 losses Ldec, Lmod, Lgrad, Llap (Eq. 16–18)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.modulo_spike_hdr.lar import lar_remainder, laplacian, spatial_gradient


@dataclass
class UnwrapLossConfig:
    lambda_mu: float = 10.0
    lambda_lin: float = 10.0
    lambda_phys: float = 1.0
    period: float = 256.0


class UnwrapLoss:
    def __init__(self, cfg: UnwrapLossConfig | None = None) -> None:
        self.cfg = cfg or UnwrapLossConfig()

    def _phys_terms(self, linear: Tensor, modulo: Tensor) -> Tensor:
        m = self.cfg.period
        l_mod = (lar_remainder(linear - modulo, m)).abs().mean()
        gx_i, gy_i = spatial_gradient(linear.unsqueeze(0) if linear.dim() == 3 else linear)
        gx_m, gy_m = spatial_gradient(modulo.unsqueeze(0) if modulo.dim() == 3 else modulo)
        rgx_i = lar_remainder(gx_i, m)
        rgy_i = lar_remainder(gy_i, m)
        rgx_m = lar_remainder(gx_m, m)
        rgy_m = lar_remainder(gy_m, m)
        l_grad = (rgx_i - rgx_m).abs().mean() + (rgy_i - rgy_m).abs().mean()
        li = linear.unsqueeze(0) if linear.dim() == 3 else linear
        lm = modulo.unsqueeze(0) if modulo.dim() == 3 else modulo
        l_lap = (lar_remainder(laplacian(li), m) - lar_remainder(laplacian(lm), m)).abs().mean()
        return l_mod + l_grad + l_lap

    def __call__(
        self,
        i_mu: Tensor,
        i_lin: Tensor,
        i_mu_gt: Tensor,
        i_lin_gt: Tensor,
        modulo: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.cfg
        l_mu = F.l1_loss(i_mu, i_mu_gt)
        l_lin = F.l1_loss(i_lin, i_lin_gt)
        l_phys = self._phys_terms(i_lin, modulo)
        loss = cfg.lambda_mu * l_mu + cfg.lambda_lin * l_lin + cfg.lambda_phys * l_phys
        return loss, {
            "loss_total": float(loss.detach()),
            "loss_mu": float(l_mu.detach()),
            "loss_lin": float(l_lin.detach()),
            "loss_phys": float(l_phys.detach()),
        }
