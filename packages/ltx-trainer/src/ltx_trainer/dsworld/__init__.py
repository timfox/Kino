"""DSWorld — data science world model for autonomous agents (arXiv:2607.15901)."""

from ltx_trainer.dsworld.benchmarks import benchmarks_bundle
from ltx_trainer.dsworld.config import (
    BENCHMARK,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    DSWorldConfig,
)
from ltx_trainer.dsworld.integration import framework_card, integration_bundle
from ltx_trainer.dsworld.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_json,
    run_reflective_demo,
    run_routing_demo,
    run_transition_demo,
)

__all__ = [
    "BENCHMARK",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "DSWorldConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_json",
    "framework_card",
    "integration_bundle",
    "run_reflective_demo",
    "run_routing_demo",
    "run_transition_demo",
]
