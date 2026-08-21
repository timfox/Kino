"""Sparse autoencoder interface (Sec. 3.2, Eq. 7)."""

from __future__ import annotations

import math


def support_indices(activations: list[float], *, eps: float = 1e-8) -> list[int]:
    r"""S(z) = {j : h_j(z) ≠ 0} (Eq. 4), toy dense list."""
    return [j for j, v in enumerate(activations) if abs(v) > eps]


def sae_reconstruction_loss(
    z_norm: float,
    z_hat_norm: float,
    h_l1: float,
    *,
    alpha: float = 0.1,
) -> float:
    r"""LSAE(z) = ||z - ẑ||²₂ + α||h(z)||₁ (Eq. 7), scalar toy."""
    return (z_norm - z_hat_norm) ** 2 + alpha * h_l1


def sparsity_fraction(active: int, total: int) -> float:
    """|S(z)| / d_sae — paper reports |S(z)| ≪ d_sae."""
    if total <= 0:
        return 0.0
    return active / total
