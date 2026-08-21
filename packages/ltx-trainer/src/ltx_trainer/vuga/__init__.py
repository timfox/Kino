"""VUGA: viewport-unaware BOIQA / unified BIQA (arXiv:2604.23953)."""

from ltx_trainer.vuga.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.vuga.config import CODE_URL, IQA_DATABASES, OIQA_DATABASES, VUGAConfig
from ltx_trainer.vuga.paper import evaluation_demo, framework_card
from ltx_trainer.vuga.pipeline import evaluation_demo_run, train_step
from ltx_trainer.vuga.vuga_net import VUGA

__all__ = [
    "CODE_URL",
    "IQA_DATABASES",
    "OIQA_DATABASES",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "VUGA",
    "VUGAConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "train_step",
]
