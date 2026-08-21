"""PanoWorld-X — sphere-aware explorable panoramic video diffusion (arXiv:2509.24997)."""

from ltx_trainer.panoworld_x.benchmarks import benchmarks_bundle
from ltx_trainer.panoworld_x.config import PAPER_ARXIV, PAPER_TITLE, PanoWorldXConfig
from ltx_trainer.panoworld_x.mock import evaluation_smoke
from ltx_trainer.panoworld_x.paper import evaluation_demo, framework_card
from ltx_trainer.panoworld_x.panoworld_x_net import PanoWorldXStub
from ltx_trainer.panoworld_x.pipeline import evaluation_demo_run

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PanoWorldXConfig",
    "PanoWorldXStub",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
]
