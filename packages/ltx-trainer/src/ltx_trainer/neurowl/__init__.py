"""NeurOWL — incomplete OWL subsumption verification + abduction (arXiv:2607.15776)."""

from ltx_trainer.neurowl.benchmarks import benchmarks_bundle
from ltx_trainer.neurowl.config import (
    BENCHMARK,
    NeurOWLConfig,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.neurowl.integration import framework_card, integration_bundle
from ltx_trainer.neurowl.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_fix,
    reason_subsumption,
)

__all__ = [
    "BENCHMARK",
    "NeurOWLConfig",
    "PAPER_ARXIV",
    "PAPER_TITLE",
    "PAPER_URL",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_fix",
    "framework_card",
    "integration_bundle",
    "reason_subsumption",
]
