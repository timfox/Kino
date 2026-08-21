"""DiffTestGen — change-directed LLM differential testing (arXiv:2607.16024)."""

from ltx_trainer.difftestgen.benchmarks import benchmarks_bundle
from ltx_trainer.difftestgen.config import (
    BENCHMARK,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    DiffTestGenConfig,
)
from ltx_trainer.difftestgen.integration import framework_card, integration_bundle
from ltx_trainer.difftestgen.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_json,
    run_access_demo,
    run_generation_demo,
)

__all__ = [
    "BENCHMARK",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "DiffTestGenConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_json",
    "framework_card",
    "integration_bundle",
    "run_access_demo",
    "run_generation_demo",
]
