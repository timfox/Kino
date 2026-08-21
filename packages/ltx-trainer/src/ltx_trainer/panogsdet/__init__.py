"""PanoGSDet: semantic Gaussian panoramic 3D detection (arXiv:2605.14601)."""

from ltx_trainer.panogsdet.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.panogsdet.config import PanoGSDetConfig, STRUCTURED3D_CATEGORIES
from ltx_trainer.panogsdet.erp_geometry import depth_map_to_points
from ltx_trainer.panogsdet.paper import evaluation_demo, framework_card
from ltx_trainer.panogsdet.pipeline import evaluation_demo_run, train_step
from ltx_trainer.panogsdet.panogsdet_net import PanoGSDet

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PanoGSDet",
    "PanoGSDetConfig",
    "STRUCTURED3D_CATEGORIES",
    "benchmarks_bundle",
    "depth_map_to_points",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "train_step",
]
