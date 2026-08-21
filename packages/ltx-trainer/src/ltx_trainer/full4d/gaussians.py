"""4D Gaussian deformation smoke (Sec. 3.3, Eq. 11)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeformationField(nn.Module):
    """(Δμ, Δq, Δs) = F_deform(μ, t) (Eq. 11)."""

    def __init__(self, dim: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim + 4 + 3),
        )

    def forward(self, mu: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if t.numel() == mu.shape[0]:
            t_in = t.view(-1, 1).float()
        else:
            t_in = t.reshape(1, 1).expand(mu.shape[0], 1).float()
        x = torch.cat([mu, t_in], dim=-1)
        out = self.net(x)
        d_mu = out[..., :3]
        d_q = out[..., 3:7]
        d_s = out[..., 7:10]
        return d_mu, d_q, d_s


def frame_dim_concat(
    z_tgt: Tensor,
    z_proj: Tensor,
) -> Tensor:
    """x_v = [x_tgt, x_proj] along frame dim (Eq. 5)."""
    return torch.cat([z_tgt, z_proj], dim=0)
