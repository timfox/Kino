"""LatentHDR-style exposure head (FiLM residual on clean latents + scalar EV).

Reference: arXiv:2605.11115 — predict z_ev = z_base + f(z_base, phi(e)) with L_ev MSE.
Phase 2 joint training with L_diff is configured separately (not wired in stock ``LtxvTrainer`` yet).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FiLMResidualExposureHead(nn.Module):
    """Small FiLM MLP: scalar EV → scale/shift on latent channels, residual add to z_base."""

    def __init__(
        self,
        *,
        latent_channels: int,
        cond_dim: int = 128,
        num_layers: int = 2,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.residual_scale = residual_scale
        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(in_dim, cond_dim), nn.SiLU()]
            in_dim = cond_dim
        layers.append(nn.Linear(in_dim, latent_channels * 2))
        self.ev_mlp = nn.Sequential(*layers)

    def forward(self, z_base: Tensor, ev: Tensor) -> Tensor:
        """
        Args:
            z_base: ``[B, C, F, H, W]`` or ``[C, F, H, W]``
            ev: ``[B]`` or scalar tensor of exposure values (EV)
        """
        if z_base.dim() == 4:
            z_base = z_base.unsqueeze(0)
            ev = ev.reshape(1) if ev.ndim == 0 else ev.reshape(1)
            squeeze = True
        else:
            squeeze = False

        if ev.ndim == 0:
            ev = ev.reshape(1)
        if ev.ndim == 1 and ev.shape[0] == 1 and z_base.shape[0] > 1:
            ev = ev.expand(z_base.shape[0])

        b, c, f, h, w = z_base.shape
        film = self.ev_mlp(ev.to(dtype=z_base.dtype, device=z_base.device).reshape(b, 1))
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(b, c, 1, 1, 1)
        beta = beta.view(b, c, 1, 1, 1)
        delta = gamma * z_base + beta
        out = z_base + self.residual_scale * delta
        return out.squeeze(0) if squeeze else out


def exposure_latent_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Scalar MSE for exposure latent supervision."""
    return torch.nn.functional.mse_loss(pred, target)
