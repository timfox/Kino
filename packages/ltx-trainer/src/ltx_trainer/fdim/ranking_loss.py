"""Pairwise ranking fidelity loss (Eq. 19–21)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _phi(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class RankingLoss(nn.Module):
    def forward(
        self,
        qi: Tensor,
        qj: Tensor,
        si: Tensor,
        sj: Tensor,
        mu_i: Tensor,
        mu_j: Tensor,
        sig_i: Tensor,
        sig_j: Tensor,
    ) -> Tensor:
        denom = torch.sqrt(sig_i * sig_i + sig_j * sig_j + 1e-6)
        g_ij = _phi((mu_i - mu_j) / denom)
        sden = torch.sqrt(si * si + sj * sj + 1e-6)
        p_ij = _phi((qi - qj) / sden)
        return (1.0 - g_ij * p_ij - (1.0 - g_ij) * (1.0 - p_ij)).mean()
