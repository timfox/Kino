"""Runtime configuration for latent PRM guidance demos."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.latent_prm_guidance.constants import (
    BRANCHES_TEST,
    LATENT_STEPS_INFER,
    PRIMARY_MODEL,
    PRM_BACKBONE,
)


@dataclass
class LatentPrmGuidanceConfig:
    primary_model: str = PRIMARY_MODEL
    prm_backbone: str = PRM_BACKBONE
    latent_steps: int = LATENT_STEPS_INFER
    branches_test: int = BRANCHES_TEST
    use_repair: bool = False
