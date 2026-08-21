"""PagedWeight — dynamic quality-aware MoE weight quantization (arXiv:2607.16184)."""

from ltx_trainer.pagedweight.benchmarks import benchmarks_bundle
from ltx_trainer.pagedweight.config import (
    BENCHMARK,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    PagedWeightConfig,
)
from ltx_trainer.pagedweight.integration import framework_card, integration_bundle
from ltx_trainer.pagedweight.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    evaluation_smoke_json,
    run_serving_demo,
)

__all__ = [
    "BENCHMARK",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "PagedWeightConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "evaluation_smoke_json",
    "framework_card",
    "integration_bundle",
    "run_serving_demo",
]
