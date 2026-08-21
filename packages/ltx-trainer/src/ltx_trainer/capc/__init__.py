"""CAPC — Cache-Aware Prompt Compression (arXiv:2607.15516)."""

from ltx_trainer.capc.baselines import (
    LONGBENCH_DOMINANCE,
    PAPER_ANCHORS,
    RHO_CROSS_TABLE,
    TAU_BENCH_RETAIL,
    benchmarks_bundle,
)
from ltx_trainer.capc.boundary import (
    AdaptiveCacheBoundary,
    SegmentClass,
    adaptive_vs_naive_savings,
    normalize,
    synthetic_version_drift,
)
from ltx_trainer.capc.compress import compress_query_agnostic, estimate_tokens
from ltx_trainer.capc.config import (
    HOT_TIER_TOKENS,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    RHO_HOT,
    CapcConfig,
)
from ltx_trainer.capc.cost_model import (
    rho_cross,
    rho_empirical,
    strategy_costs,
    tier_preserving_rmax,
    tier_preserving_rmax_chars,
)
from ltx_trainer.capc.mock import evaluation_smoke
from ltx_trainer.capc.pipeline import (
    evaluation_demo,
    framework_card,
    knowledge_card,
)

__all__ = [
    "AdaptiveCacheBoundary",
    "HOT_TIER_TOKENS",
    "LONGBENCH_DOMINANCE",
    "PAPER_ANCHORS",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "RHO_CROSS_TABLE",
    "RHO_HOT",
    "SegmentClass",
    "TAU_BENCH_RETAIL",
    "CapcConfig",
    "adaptive_vs_naive_savings",
    "benchmarks_bundle",
    "compress_query_agnostic",
    "estimate_tokens",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "normalize",
    "rho_cross",
    "rho_empirical",
    "strategy_costs",
    "synthetic_version_drift",
    "tier_preserving_rmax",
    "tier_preserving_rmax_chars",
]
