"""360Anything: geometry-free perspective→360° (arXiv:2601.16192)."""

from ltx_trainer.anything360.anything360_model import Anything360Stub
from ltx_trainer.anything360.benchmarks import benchmarks_bundle, table1_laval
from ltx_trainer.anything360.config import Anything360Config, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.anything360.mock import evaluation_smoke
from ltx_trainer.anything360.paper import evaluation_demo, framework_card
from ltx_trainer.anything360.pipeline import evaluation_demo_run, train_step

__all__ = [
    "Anything360Config",
    "Anything360Stub",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
    "table1_laval",
    "train_step",
]
