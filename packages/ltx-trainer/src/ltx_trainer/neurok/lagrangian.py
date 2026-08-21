"""Lagrangian simulation in NEUROK latent space (Sec. 5)."""

from __future__ import annotations

import torch
from torch import Tensor


def kinetic_energy(z_dot: Tensor, mass: float | Tensor = 1.0) -> Tensor:
    """T = ½ m ||ż||² (quadratic kinetic term)."""
    m = mass if isinstance(mass, Tensor) else torch.tensor(mass, device=z_dot.device, dtype=z_dot.dtype)
    return 0.5 * m * (z_dot * z_dot).sum()


def potential_energy(z: Tensor, *, stiffness: float = 0.5) -> Tensor:
    """Harmonic well V(z) = ½ k ||z||² around rest configuration."""
    return 0.5 * stiffness * (z * z).sum()


def lagrangian(z: Tensor, z_dot: Tensor, *, mass: float = 1.0, stiffness: float = 0.5) -> Tensor:
    """L(z, ż) = T − V."""
    return kinetic_energy(z_dot, mass) - potential_energy(z, stiffness=stiffness)


def metric_from_decoder_jacobian(decoder_weight: Tensor, z: Tensor) -> Tensor:
    """G(z) = J_z^T J_z with linear decoder F(z) = W z."""
    j = decoder_weight  # (out, in)
    g = j.T @ j
    return g + 1e-4 * torch.eye(g.shape[0], device=g.device, dtype=g.dtype)


def euler_lagrange_step(
    z: Tensor,
    z_dot: Tensor,
    *,
    decoder_weight: Tensor,
    mass: float = 1.0,
    stiffness: float = 0.5,
    dt: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """Semi-implicit step for m G(z) z̈ + k z ≈ 0 (simplified Eq. 3)."""
    g = metric_from_decoder_jacobian(decoder_weight, z)
    force = -stiffness * z
    accel = torch.linalg.solve(g, force.unsqueeze(-1)).squeeze(-1) / mass
    z_dot_new = z_dot + dt * accel
    z_new = z + dt * z_dot_new
    return z_new, z_dot_new


def simulate_latent_trajectory(
    z0: Tensor,
    z_dot0: Tensor,
    *,
    decoder_weight: Tensor,
    steps: int,
    dt: float = 0.02,
    mass: float = 1.0,
    stiffness: float = 0.5,
) -> list[Tensor]:
    traj = [z0]
    z, z_dot = z0, z_dot0
    for _ in range(steps):
        z, z_dot = euler_lagrange_step(
            z,
            z_dot,
            decoder_weight=decoder_weight,
            mass=mass,
            stiffness=stiffness,
            dt=dt,
        )
        traj.append(z.detach().clone())
    return traj


def total_energy(z: Tensor, z_dot: Tensor, *, mass: float = 1.0, stiffness: float = 0.5) -> float:
    return float((kinetic_energy(z_dot, mass) + potential_energy(z, stiffness=stiffness)).item())
