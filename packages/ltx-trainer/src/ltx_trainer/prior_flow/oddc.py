"""Ortho-Driven Distortion Compensation (Sec. 3.4, Eq. 12–16)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.prior_flow.orthogonal_view import flow_orthogonal_to_primitive


def group_wise_correlation(f1: Tensor, f2_warped: Tensor, groups: int = 8) -> Tensor:
    """GW confidence stub (Eq. 13)."""
    b, c, h, w = f1.shape
    g = min(groups, c)
    f1g = f1.view(b, g, c // g, h, w)
    f2g = f2_warped.view(b, g, c // g, h, w)
    return (f1g * f2g).mean(dim=2)


class ODDCEncoder(nn.Module):
    """Confidence-guided motion encoder (Enc, Eng, Enf stub)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.enc = nn.Conv2d(2, dim, 3, padding=1)
        self.eng = nn.Conv2d(2 * 8, dim, 3, padding=1)
        self.enf = nn.Conv2d(4, dim, 3, padding=1)
        self.fuse = nn.Conv2d(dim * 3, dim, 3, padding=1)

    def forward(
        self,
        cp: Tensor,
        co2p: Tensor,
        gp: Tensor,
        go2p: Tensor,
        fp: Tensor,
        fo2p: Tensor,
    ) -> Tensor:
        m_corr = self.enc(torch.cat([cp, co2p], dim=1))
        m_conf = self.eng(torch.cat([gp, go2p], dim=1))
        m_flow = self.enf(torch.cat([fp, fo2p], dim=1))
        return self.fuse(torch.cat([m_corr, m_conf, m_flow], dim=1))
