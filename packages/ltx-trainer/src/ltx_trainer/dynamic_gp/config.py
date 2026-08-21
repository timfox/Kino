"""Dynamic Gaussian Process configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicGPConfig:
    num_basis: int = 31
    state_dim: int = 1
    alpha_diffusivity: float = 0.1
    dt: float = 0.1
    sigma_w: float = 0.316
    sigma_v: float = 0.316
    domain_lo: float = -1.0
    domain_hi: float = 1.0
    num_steps: int = 5
    num_obs: int = 5
