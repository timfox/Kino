"""VarRate — training-free variable-rank KV coding (arXiv:2607.15498)."""

from ltx_trainer.varrate.allocate import allocate_for_sequence, snapkv_salience, water_fill_ranks
from ltx_trainer.varrate.baselines import (
    PAPER_ANCHORS,
    TABLE_1_TWO_MODEL,
    TABLE_2_REUSE,
    TABLE_3_PREFILL,
    benchmarks_bundle,
)
from ltx_trainer.varrate.config import (
    KAPPA,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    R_MAX,
    RMIN,
    VarRateConfig,
)
from ltx_trainer.varrate.mock import evaluation_smoke
from ltx_trainer.varrate.pipeline import evaluation_demo, framework_card, knowledge_card

__all__ = [
    "KAPPA",
    "PAPER_ANCHORS",
    "PAPER_ARXIV",
    "PAPER_SYSTEM",
    "PAPER_TITLE",
    "PAPER_URL",
    "RMIN",
    "R_MAX",
    "TABLE_1_TWO_MODEL",
    "TABLE_2_REUSE",
    "TABLE_3_PREFILL",
    "VarRateConfig",
    "allocate_for_sequence",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "knowledge_card",
    "snapkv_salience",
    "water_fill_ranks",
]
