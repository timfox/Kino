"""ErA configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.era_defocus.constants import (
    KERNEL_SIZE,
    PATCH_SIZE,
    TRAIN_DEFAULTS,
    UNROLLING_DEPTH_K,
)


@dataclass
class EraDefocusConfig:
    unrolling_depth: int = UNROLLING_DEPTH_K
    kernel_size: int = KERNEL_SIZE
    patch_size: int = PATCH_SIZE
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    omega_recon: float = float(TRAIN_DEFAULTS["omega_recon"])
    temperature: float = 1.0
