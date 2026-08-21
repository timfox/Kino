"""Legs Over Arms: lower-body pose for HST trajectory prediction (arXiv:2602.09076)."""

from ltx_trainer.legs_over_arms.benchmarks import benchmarks_bundle, table1_best_lower_body
from ltx_trainer.legs_over_arms.config import PAPER_ARXIV, LegsOverArmsConfig
from ltx_trainer.legs_over_arms.datasets import datasets_bundle
from ltx_trainer.legs_over_arms.hst_net import HSTStub
from ltx_trainer.legs_over_arms.paper import evaluation_demo, framework_card
from ltx_trainer.legs_over_arms.pipeline import evaluation_demo_run, train_step

__all__ = [
    "PAPER_ARXIV",
    "HSTStub",
    "LegsOverArmsConfig",
    "benchmarks_bundle",
    "datasets_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "table1_best_lower_body",
    "train_step",
]
