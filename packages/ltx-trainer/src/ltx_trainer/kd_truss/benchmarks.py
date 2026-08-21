"""Paper tables and anchors (Hu et al., arXiv:2606.11582)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.kd_truss.config import (
    DEFAULT_DELTA_FRAC,
    DEFAULT_K_FRAC,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
)

# Table I — dataset statistics (Sec. VII-A)
TABLE_I_DATASETS: list[dict[str, Any]] = [
    {"dataset": "Email", "V": 900, "E": 16000, "n": 803, "tau": 11.5, "triangles": 105000, "kmax": 23, "delta_max": 800},
    {"dataset": "Mathoverflow", "V": 24000, "E": 187000, "n": 2450, "tau": 1.6, "triangles": 1400000, "kmax": 42, "delta_max": 2336},
    {"dataset": "Askubuntu", "V": 159000, "E": 455000, "n": 2613, "tau": 1.2, "triangles": 680000, "kmax": 26, "delta_max": 2040},
    {"dataset": "Superuser", "V": 194000, "E": 714000, "n": 2773, "tau": 1.2, "triangles": 1500000, "kmax": 35, "delta_max": 2692},
    {"dataset": "Wikitalk", "V": 1100000, "E": 2700000, "n": 2320, "tau": 1.4, "triangles": 8100000, "kmax": 49, "delta_max": 2231},
    {"dataset": "Youtube", "V": 322000, "E": 9300000, "n": 225, "tau": 1.0, "triangles": 12000000, "kmax": 33, "delta_max": 225},
    {"dataset": "Stackoverflow", "V": 2600000, "E": 28100000, "n": 2774, "tau": 1.2, "triangles": 114200000, "kmax": 79, "delta_max": 2768},
    {"dataset": "Wikipedia", "V": 1800000, "E": 36500000, "n": 2235, "tau": 1.1, "triangles": 126600000, "kmax": 59, "delta_max": 2231},
]

# Table II — index compression (Sec. VII-C)
TABLE_II_INDEX_STATS: list[dict[str, Any]] = [
    {"dataset": "Email", "avg_kspan": 290, "tc_edges": 162000, "dc_edges": 154000, "ratio": 9.57, "compression": 17.5e-4},
    {"dataset": "MathOverflow", "avg_kspan": 1478, "tc_edges": 1959000, "dc_edges": 1871000, "ratio": 10.40, "compression": 6.25e-4},
    {"dataset": "Wikipedia", "avg_kspan": 1304, "tc_edges": 164240000, "dc_edges": 163400000, "ratio": 4.47, "compression": 8.60e-4},
]

# Email case study Fig. 2 (k=16)
TABLE_EMAIL_CASE: list[dict[str, Any]] = [
    {"delta": "inf", "vertices": 213, "edges": 4402, "triangles": 42683, "clustering": 0.72},
    {"delta": 200, "vertices": 130, "edges": 2355, "triangles": 21738, "clustering": 0.77},
    {"delta": 150, "vertices": 108, "edges": 1735, "triangles": 14978, "clustering": 0.81},
    {"delta": 100, "vertices": 38, "edges": 564, "triangles": 4670, "clustering": 0.85},
]

PAPER_ANCHORS = {
    "datasets": len(TABLE_I_DATASETS),
    "index_speedup_orders": "2-4",
    "max_compression_ratio": 1.0e-4,
    "default_k_frac": DEFAULT_K_FRAC,
    "default_delta_frac": DEFAULT_DELTA_FRAC,
    "query_ms_wikipedia_max": 410,
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
        "table_ii_index": TABLE_II_INDEX_STATS,
        "table_email_case": TABLE_EMAIL_CASE,
        "anchors": PAPER_ANCHORS,
    }
