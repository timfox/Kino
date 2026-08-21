"""AgentFAIR — multi-agent geospatial FAIRness evaluation (arXiv:2607.15781)."""

from ltx_trainer.agentfair.benchmarks import benchmarks_bundle
from ltx_trainer.agentfair.config import (
    BENCHMARK,
    AgentFAIRConfig,
    PAPER_ARXIV,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    SUB_PRINCIPLES,
)
from ltx_trainer.agentfair.integration import framework_card, integration_bundle
from ltx_trainer.agentfair.pipeline import (
    evaluate_dataset,
    evaluate_demo_suite,
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_fix,
)

__all__ = [
    "BENCHMARK",
    "AgentFAIRConfig",
    "PAPER_ARXIV",
    "PAPER_REPO",
    "PAPER_TITLE",
    "PAPER_URL",
    "SUB_PRINCIPLES",
    "benchmarks_bundle",
    "evaluate_dataset",
    "evaluate_demo_suite",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_fix",
    "framework_card",
    "integration_bundle",
]
