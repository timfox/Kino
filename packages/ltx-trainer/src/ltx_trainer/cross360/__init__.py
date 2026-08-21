"""Cross360: 360° depth via cross projections (arXiv:2601.17271)."""

from ltx_trainer.cross360.benchmarks import benchmarks_bundle, table1_ours_m3d, table2_ours_struct3d
from ltx_trainer.cross360.config import CODE_URL, Cross360Config, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.cross360.cross360_net import Cross360NetStub
from ltx_trainer.cross360.mock import evaluation_smoke
from ltx_trainer.cross360.paper import evaluation_demo, framework_card
from ltx_trainer.cross360.pipeline import evaluation_demo_run, train_step

__all__ = [
    "CODE_URL",
    "Cross360Config",
    "Cross360NetStub",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
    "table1_ours_m3d",
    "table2_ours_struct3d",
    "train_step",
]
