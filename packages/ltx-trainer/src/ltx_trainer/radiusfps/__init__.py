"""RadiusFPS spherical-voxel FPS acceleration (Yu et al., arXiv:2606.06255)."""

from ltx_trainer.radiusfps.config import RadiusFpsConfig
from ltx_trainer.radiusfps.mock import evaluation_smoke
from ltx_trainer.radiusfps.paper import knowledge_bundle, paper_card
from ltx_trainer.radiusfps.pipeline import run_demo

__all__ = [
    "RadiusFpsConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
