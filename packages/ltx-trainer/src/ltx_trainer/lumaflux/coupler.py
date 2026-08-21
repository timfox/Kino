"""HDR Residual Coupler (Sec. 4.5, Eq. 15)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HDRResidualCoupler(nn.Module):
    """z_out = z_res + λ(Wp T_phys + Wc Cperc(T_perc))."""

    def __init__(self, token_dim: int, phys_channels: int = 32, perc_dim: int = 64) -> None:
        super().__init__()
        self.wp = nn.Linear(phys_channels, token_dim)
        self.wc = nn.Linear(perc_dim, token_dim)

    def forward(
        self,
        z_res: Tensor,
        t_phys: Tensor,
        t_perc: Tensor,
        mod: dict[str, Tensor],
    ) -> Tensor:
        b, n, d = z_res.shape
        phys_pool = t_phys.mean(dim=(-2, -1))
        perc_pool = t_perc.mean(dim=1)
        coupling = self.wp(phys_pool).unsqueeze(1) + self.wc(perc_pool).unsqueeze(1)
        lam = mod["lambda_t"].unsqueeze(1)
        return z_res + lam * coupling
