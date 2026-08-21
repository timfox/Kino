"""Physics-guided UNet+FNO flood prediction (Gebre et al., arXiv:2606.06524)."""

from ltx_trainer.flood_physics.config import FloodPhysicsConfig
from ltx_trainer.flood_physics.mock import evaluation_smoke
from ltx_trainer.flood_physics.paper import knowledge_bundle, paper_card
from ltx_trainer.flood_physics.pipeline import run_demo

__all__ = [
    "FloodPhysicsConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
