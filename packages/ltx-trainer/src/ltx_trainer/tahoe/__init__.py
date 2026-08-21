"""Tahoe — Text-to-SQL Hint Bank (arXiv:2606.12387)."""

from ltx_trainer.tahoe.config import (
    BENCHMARK,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    TahoeConfig,
)
from ltx_trainer.tahoe.benchmarks import benchmarks_bundle
from ltx_trainer.tahoe.deployment import classify_deployment_batch
from ltx_trainer.tahoe.development import run_development_batch
from ltx_trainer.tahoe.hints import HintBank
from ltx_trainer.tahoe.inference import hint_guided_inference
from ltx_trainer.tahoe.integration import framework_card, integration_bundle
from ltx_trainer.tahoe.pipeline import evaluation_demo, evaluation_smoke
from ltx_trainer.tahoe.query_pipeline import run_hint_guided_query
from ltx_trainer.tahoe.running_example import seed_hint_bank
from ltx_trainer.tahoe.strategy_attribution import run_strategy_attribution

__all__ = [
    "BENCHMARK",
    "HintBank",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "TahoeConfig",
    "benchmarks_bundle",
    "classify_deployment_batch",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "hint_guided_inference",
    "integration_bundle",
    "run_development_batch",
    "run_hint_guided_query",
    "run_strategy_attribution",
    "seed_hint_bank",
]
