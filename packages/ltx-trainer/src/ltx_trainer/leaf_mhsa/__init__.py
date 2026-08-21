"""Grapevine leaf trait → reflectance MHSA (SPIE 2025, 10.1117/12.3061298)."""

from ltx_trainer.leaf_mhsa.config import LeafMHSAConfig
from ltx_trainer.leaf_mhsa.mock import evaluation_smoke
from ltx_trainer.leaf_mhsa.pipeline import evaluation_demo, framework_card
from ltx_trainer.leaf_mhsa.ltx_plan import ltx_integration_plan

__all__ = [
    "LeafMHSAConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
