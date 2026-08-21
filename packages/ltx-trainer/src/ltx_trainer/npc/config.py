"""Hyperparameters for finite-blocklength noisy permutation channel bounds (Feng et al., arXiv:2605.25699)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NPCConfig:
    """Defaults for strictly positive DMC ``W`` with affine output dimension ``d``."""

    epsilon: float = 1e-3
    """Target average error probability ϵ."""

    lattice_resolution: int | None = None
    """Simplex lattice resolution ``N``; if None, use ``floor(c * sqrt(n))`` in Gaussian mode."""

    gaussian_c: float | None = None
    """Scale ``c`` in ``N = floor(c sqrt(n))`` for achievability (Sec. V)."""

    covering_a: float = 1.0
    """KL covering radius parameter ``a`` in ``ρ = a²/n`` (Sec. IV)."""

    meta_converse_eta_margin: float = 0.0
    """η in (ϵ, 1) for modified meta-converse; paper uses η = (1+ϵ)/2."""
