"""Shared/unique factorization and logit injection (Sec. 3.2–3.3, Eq. 5, 8, 9)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GatedFactorEncoder(nn.Module):
    """Gated MLP factor head (Eq. 5): ``z = g ⊙ ϕ(h)``, ``g = σ(W h)``."""

    def __init__(self, hidden_dim: int, factor_dim: int, mlp_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = hidden_dim
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.GELU()])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, factor_dim))
        self.phi = nn.Sequential(*layers)
        self.gate = nn.Linear(hidden_dim, factor_dim)
        self.ln = nn.LayerNorm(factor_dim)

    def forward(self, h: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(h))
        return self.ln(g * self.phi(h))


class DualFactorization(nn.Module):
    """Shared + unique encoders for one flow (understanding or generation)."""

    def __init__(self, hidden_dim: int, factor_dim: int) -> None:
        super().__init__()
        self.shared = GatedFactorEncoder(hidden_dim, factor_dim)
        self.unique = GatedFactorEncoder(hidden_dim, factor_dim)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.shared(h), self.unique(h)


def orthogonality_loss(z_sh: Tensor, z_uni: Tensor) -> Tensor:
    r"""``L⊥ = Σ_i ‖(z_sh^i)ᵀ z_uni^i‖²_F`` (Eq. 9), batch-mean over samples."""
    if z_sh.dim() == 1:
        z_sh = z_sh.unsqueeze(0)
        z_uni = z_uni.unsqueeze(0)
    # Per-sample inner product squared, summed over factor dims
    inner = (z_sh * z_uni).sum(dim=-1).pow(2)
    return inner.mean()


def inject_logits(
    s_u: Tensor,
    s_g: Tensor,
    z_g_sh: Tensor,
    z_u_uni: Tensor,
    z_u_sh: Tensor,
    z_g_uni: Tensor,
    a_u: Tensor,
    a_g: Tensor,
    b_u: Tensor,
    b_g: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Cross-task logit bias injection (Eq. 8).

    ``\tilde{s}_U = s_U + A_U z_G^sh + B_U z_U^uni``,
    ``\tilde{s}_G = s_G + A_G z_U^sh + B_G z_G^uni``.
    Readout matrices ``A_*``, ``B_*`` are low-rank in training; here they are caller-supplied projections.
    """
    tilde_u = s_u + a_u @ z_g_sh + b_u @ z_u_uni
    tilde_g = s_g + a_g @ z_u_sh + b_g @ z_g_uni
    return tilde_u, tilde_g


def pool_image_tokens(h_img: Tensor) -> Tensor:
    """Mean-pool image-token hidden states ``[N, D] → [D]`` (Sec. 3.2)."""
    if h_img.dim() == 1:
        return h_img
    return h_img.mean(dim=0)
