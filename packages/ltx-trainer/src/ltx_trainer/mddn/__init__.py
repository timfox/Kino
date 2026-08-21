"""MDDN: multi-level distortion-aware deformable ODISR (arXiv:2512.17343)."""

from ltx_trainer.mddn.benchmarks import benchmarks_bundle
from ltx_trainer.mddn.config import MddnConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.mddn.distortion import erp_distortion_map
from ltx_trainer.mddn.mddn_net import MDDNStub
from ltx_trainer.mddn.mock import evaluation_smoke
from ltx_trainer.mddn.paper import evaluation_demo, framework_card
from ltx_trainer.mddn.pipeline import evaluation_demo_run, train_step

__all__ = [
    "MDDNStub",
    "MddnConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "erp_distortion_map",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
    "train_step",
]
