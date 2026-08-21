"""Flood physics-guided DL configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FloodPhysicsConfig:
    lambda_ext: float = 1.0
    lambda_h: float = 1.0
    lambda_u: float = 0.5
    lambda_v: float = 0.5
    lambda_phys: float = 0.1
    lambda_reg: float = 1e-4
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    manning_n: float = 0.035
    gravity: float = 9.81
    fno_modes: int = 16
    unet_channels: int = 64
