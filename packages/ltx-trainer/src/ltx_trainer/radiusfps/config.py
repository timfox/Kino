"""RadiusFPS demo configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RadiusFpsConfig:
    num_samples: int = 32
    nvox: int = 16
    seed_index: int = 0
    variant: str = "radiusfps"  # standard | radiusfps
