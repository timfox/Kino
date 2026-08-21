"""OB3D — Omnidirectional Blender 3D reconstruction benchmark."""

from ltx_trainer.ob3d.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, OB3DConfig
from ltx_trainer.ob3d.mock import evaluation_smoke

__all__ = [
    "OB3DConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "evaluation_smoke",
]
