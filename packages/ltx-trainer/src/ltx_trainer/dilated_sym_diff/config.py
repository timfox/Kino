"""Dilated symmetric difference configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DilatedSymDiffConfig:
    default_radius: int = 8
    delta_align: float = 7.5
    min_feature_dim: int = 16
    learnable_radius: bool = False
