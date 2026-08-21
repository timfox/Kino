"""MAS-IB — multi-agent vs single-agent information bottleneck (arXiv:2607.16133)."""

from ltx_trainer.mas_ib.benchmarks import benchmarks_bundle
from ltx_trainer.mas_ib.config import (
    BENCHMARK,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    MasIbConfig,
)
from ltx_trainer.mas_ib.integration import framework_card, integration_bundle
from ltx_trainer.mas_ib.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_json,
    run_regime_demo,
)

__all__ = [
    "BENCHMARK",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "MasIbConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_json",
    "framework_card",
    "integration_bundle",
    "run_regime_demo",
]
