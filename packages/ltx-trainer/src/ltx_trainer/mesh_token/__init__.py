"""MeshToken render-free 3D mesh motion control (arXiv:2606.02000)."""

from ltx_trainer.mesh_token.config import MeshTokenConfig
from ltx_trainer.mesh_token.mock import evaluation_smoke
from ltx_trainer.mesh_token.pipeline import evaluation_demo, framework_card
from ltx_trainer.mesh_token.ltx_plan import ltx_integration_plan

__all__ = [
    "MeshTokenConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
