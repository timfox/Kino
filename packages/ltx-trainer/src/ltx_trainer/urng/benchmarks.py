"""Paper tables and experiment anchors (Liang et al., arXiv:2606.11789)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.urng.config import (
    DEFAULT_EF_ATTRIBUTE,
    DEFAULT_EF_SPATIAL,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EDGES_IF,
    DEFAULT_MAX_EDGES_IS,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
)

# Table 1 — dataset statistics (Sec. 5.1)
TABLE_I_DATASETS: list[dict[str, Any]] = [
    {"dataset": "DB-OpenAI", "D": 990000, "Q": 10000, "d": 1536, "vector_type": "text"},
    {"dataset": "GIST1M", "D": 1000000, "Q": 1000, "d": 960, "vector_type": "image"},
    {"dataset": "S&P 500", "D": 1445794, "Q": 14603, "d": 384, "vector_type": "financial"},
    {"dataset": "SIFT1M", "D": 1000000, "Q": 10000, "d": 128, "vector_type": "image"},
    {"dataset": "DEEP1M", "D": 990000, "Q": 10000, "d": 96, "vector_type": "image"},
]

# Exp-1 / Exp-2 headline anchors (Sec. 5.2)
TABLE_IFANN_SPEEDUP: list[dict[str, Any]] = [
    {
        "dataset": "S&P 500",
        "recall_at_10": 0.95,
        "ug_qps": 62029,
        "best_baseline_qps": 587,
        "speedup": 105,
        "baseline": "H-HCNNG",
    },
    {
        "dataset": "S&P 500",
        "recall_at_10": 0.98,
        "ug_qps": 40000,
        "best_baseline_qps": 136,
        "speedup": 294,
        "baseline": "H-NSG",
    },
]

TABLE_ISANN_SPEEDUP: list[dict[str, Any]] = [
    {"dataset": "GIST1M", "recall_at_10": 0.95, "ug_qps": 2127.66, "hnsw_qps": 245, "speedup": 8.7},
    {"dataset": "SIFT1M", "recall_at_10": 0.95, "ug_qps": 33333, "hnsw_qps": 3676, "speedup": 9.1},
]

TABLE_INDEX_COST_GIST: list[dict[str, Any]] = [
    {"method": "UG", "build_s": 678.9, "index_mb": 218.7},
    {"method": "H-NSG", "build_s": 947.5, "index_mb": None},
    {"method": "H-HNSW", "build_s": None, "index_mb": 262.0},
    {"method": "H-HCNNG", "build_s": None, "index_mb": 272.4},
]

PAPER_ANCHORS = {
    "datasets": len(TABLE_I_DATASETS),
    "query_types": ["IFANN", "ISANN", "RFANN", "RSANN"],
    "urng_overhead_constant": "C_urng ≈ 31/3",
    "exact_urng_build": "O(n^3)",
    "ug_build": "O(T·C·M_ug)",
    "default_ef_spatial": DEFAULT_EF_SPATIAL,
    "default_ef_attribute": DEFAULT_EF_ATTRIBUTE,
    "default_max_edges": DEFAULT_MAX_EDGES_IF,
    "default_iterations": DEFAULT_ITERATIONS,
    "max_edges_is": DEFAULT_MAX_EDGES_IS,
    "scalability_max_m": 40_000_000,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "table_i_datasets": TABLE_I_DATASETS,
        "table_ifann_speedup": TABLE_IFANN_SPEEDUP,
        "table_isann_speedup": TABLE_ISANN_SPEEDUP,
        "table_index_cost_gist": TABLE_INDEX_COST_GIST,
        "anchors": PAPER_ANCHORS,
    }
