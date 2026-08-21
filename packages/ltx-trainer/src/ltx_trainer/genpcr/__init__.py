"""Generative point cloud registration (arXiv:2512.09407)."""

from ltx_trainer.genpcr.benchmarks import benchmarks_bundle
from ltx_trainer.genpcr.config import GenPcrConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.genpcr.match_controlnet import MatchControlNetStub
from ltx_trainer.genpcr.mock import evaluation_smoke
from ltx_trainer.genpcr.paper import evaluation_demo, framework_card
from ltx_trainer.genpcr.pipeline import evaluation_demo_run
from ltx_trainer.genpcr.registration_stub import GenerativePcrStub

__all__ = [
    "GenPcrConfig",
    "GenerativePcrStub",
    "MatchControlNetStub",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
]
