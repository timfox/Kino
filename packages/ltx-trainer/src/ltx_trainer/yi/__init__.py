"""Yi — in-place graph-based vector index updates (arXiv:2607.15576)."""

from ltx_trainer.yi.benchmarks import benchmarks_bundle
from ltx_trainer.yi.config import BENCHMARK, PAPER_ARXIV, PAPER_SYSTEM, PAPER_TITLE, PAPER_URL, YiConfig
from ltx_trainer.yi.integration import framework_card, integration_bundle
from ltx_trainer.yi.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_json,
    run_update_demo,
)

__all__ = [
    "BENCHMARK",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "YiConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_json",
    "framework_card",
    "integration_bundle",
    "run_update_demo",
]
