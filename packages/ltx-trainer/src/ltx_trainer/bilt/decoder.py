"""Physics-constrained linear decoder (Sec. 2.2.1, Layer 17–18)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.bilt.config import DECODER_OUT, LATENT_DIM, SPECTRAL_POINTS


class PhysicsConstrainedDecoder(nn.Module):
    """
    Non-negative linear mixture with absorption/scattering decoupling.

    Neuron 0 (scatterer) → μ'_s at odd indices; absorbers → μ_a at even indices.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, out_dim: int = DECODER_OUT) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(latent_dim, out_dim))
        self._build_mask(out_dim)

    def _build_mask(self, out_dim: int) -> None:
        mask = torch.zeros(LATENT_DIM, out_dim)
        for i in range(SPECTRAL_POINTS):
            mask[0, 2 * i + 1] = 1.0  # scatterer → μ'_s
            mask[1, 2 * i] = 1.0  # red ink → μ_a
            mask[2, 2 * i] = 1.0  # black ink → μ_a
        self.register_buffer("mask", mask)

    def forward(self, z: Tensor) -> Tensor:
        w = torch.relu(self.weight) * self.mask
        flat = z @ w  # (B, 300)
        op = flat.view(-1, SPECTRAL_POINTS, 2)
        return op  # (:, :, 0)=μ_a interleaved handling via reshape

    @staticmethod
    def split_channels(op: Tensor) -> tuple[Tensor, Tensor]:
        """Extract μ_a and μ'_s from interleaved decoder output."""
        mu_a = op[..., 0]
        mu_s = op[..., 1]
        return mu_a, mu_s
