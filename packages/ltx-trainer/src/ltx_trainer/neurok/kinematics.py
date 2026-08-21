"""Kinematic state parameterization (Def. 1–2, Sec. 3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class KinematicParameterization:
    """Pair (Z, F): latent state space and decoder to vertex configuration."""

    latent_dim: int
    num_vertices: int

    def decode(self, z: Tensor, decoder: Callable[[Tensor], Tensor]) -> Tensor:
        """Map z ∈ Z to flattened vertex positions x ∈ R^{3n}."""
        if z.ndim != 1:
            raise ValueError("z must be a 1-D latent vector")
        return decoder(z)

    def latent_shape(self, batch: int = 1) -> tuple[int, ...]:
        return (batch, self.latent_dim)


def vertex_dim(num_vertices: int) -> int:
    return 3 * num_vertices


def random_plausible_latent(dim: int, *, device: torch.device | None = None) -> Tensor:
    """Sample near origin of learned Gaussian prior (inference-time prior)."""
    return torch.randn(dim, device=device) * 0.1


def configuration_manifold_dim(num_vertices: int, intrinsic_dof: int | None = None) -> int:
    """Intrinsic DOF k_int ≪ 3n for plausible deformations."""
    k = intrinsic_dof or max(3, num_vertices // 4)
    return min(k, 3 * num_vertices - 1)
