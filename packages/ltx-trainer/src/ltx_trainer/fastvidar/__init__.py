"""FastViDAR — omnidirectional depth via AHA (arXiv:2509.23733)."""

from ltx_trainer.fastvidar.benchmarks import benchmarks_bundle
from ltx_trainer.fastvidar.config import PAPER_ARXIV, PAPER_TITLE, FastViDARConfig
from ltx_trainer.fastvidar.fastvidar_net import FastViDARStub
from ltx_trainer.fastvidar.mock import evaluation_smoke
from ltx_trainer.fastvidar.paper import evaluation_demo, framework_card
from ltx_trainer.fastvidar.pipeline import evaluation_demo_run

__all__ = [
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "FastViDARConfig",
    "FastViDARStub",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
]
