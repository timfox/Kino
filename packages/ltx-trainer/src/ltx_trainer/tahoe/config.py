"""Tahoe configuration (Chen et al., arXiv:2606.12387)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2606.12387"
PAPER_TITLE = (
    "Tahoe: Text-to-SQL with Automated Hint Optimization from Experience"
)
PAPER_AUTHORS = "Zhiyi Chen, Jie Song, Peng Li (ByteDance / Georgia Tech)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "Preprint Jun 2026"
BENCHMARK = "Spider 2.0–Snow-0212"
BENCHMARK_URL = "https://spider2-sql.github.io/"


@dataclass
class TahoeConfig:
    dialect: str = "snowflake"
    inference_samples: int = 4
    max_learning_iterations: int = 3
    sampling_temperature: float = 0.3
    development_examples: int = 113
    held_out_examples: int = 434
    syntax_hint_count: int = 11
    semantic_hint_count: int = 37
