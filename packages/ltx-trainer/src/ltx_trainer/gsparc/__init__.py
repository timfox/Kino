"""GSpaRC — Gaussian splatting for real-time RF channel reconstruction (arXiv:2511.22793)."""

from ltx_trainer.gsparc.benchmarks import benchmarks_bundle
from ltx_trainer.gsparc.config import GSpaRCConfig, PAPER_ARXIV, PAPER_TITLE
from ltx_trainer.gsparc.gsparc_net import GSpaRCStub
from ltx_trainer.gsparc.mock import evaluation_smoke
from ltx_trainer.gsparc.paper import evaluation_demo, framework_card
from ltx_trainer.gsparc.pipeline import evaluation_demo_run

__all__ = [
    "GSpaRCConfig",
    "GSpaRCStub",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_demo_run",
    "evaluation_smoke",
    "framework_card",
]
