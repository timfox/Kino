"""PanoAffordanceNet: holistic 360° affordance grounding (arXiv:2603.09760)."""

from ltx_trainer.pano_affordance.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.pano_affordance.config import CODE_URL, NUM_AFFORDANCE_CLASSES, PanoAffordanceConfig
from ltx_trainer.pano_affordance.pano_affordance_net import PanoAffordanceNet
from ltx_trainer.pano_affordance.paper import evaluation_demo, framework_card
from ltx_trainer.pano_affordance.pipeline import evaluation_demo_run, train_step

__all__ = [
    "CODE_URL",
    "NUM_AFFORDANCE_CLASSES",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PanoAffordanceConfig",
    "PanoAffordanceNet",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "train_step",
]
