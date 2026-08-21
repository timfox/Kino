"""Paper anchors and table stubs for NeurOWL (Yang et al., arXiv:2607.15776)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "n_datasets": 3,
    "foodona_concepts": 30995,
    "snomeda_concepts": 364352,
    "prune_rate": 0.05,
    "foodona_ont_random_f1": 0.960,
    "foodona_ont_random_xf1": 0.893,
    "foodona_ont_random_xf1_star": 0.793,
    "foodona_ont_hard_f1": 0.845,
    "foodona_ont_hard_xf1": 0.781,
    "snomeda_ont_random_f1": 0.968,
    "snomeda_ont_hard_f1": 0.862,
    "snomed_exists_ont_random_f1": 0.924,
    "snomed_exists_ont_random_xf1": 0.497,
    "top_k_default": 10,
    "llm": "Qwen3.5-9B",
}

# Table 2 excerpt — NeurOWL (OnT) rows
TABLE_2_NEUROWL_ONT: list[dict[str, Any]] = [
    {
        "dataset": "FoodOnA",
        "neg": "random",
        "f1": 0.960,
        "xf1": 0.893,
        "xf1_star": 0.793,
    },
    {
        "dataset": "FoodOnA",
        "neg": "hard",
        "f1": 0.845,
        "xf1": 0.781,
        "xf1_star": 0.684,
    },
    {
        "dataset": "SnomedA",
        "neg": "random",
        "f1": 0.968,
        "xf1": 0.847,
        "xf1_star": 0.659,
    },
    {
        "dataset": "SnomedA",
        "neg": "hard",
        "f1": 0.862,
        "xf1": 0.745,
        "xf1_star": 0.569,
    },
    {
        "dataset": "Snomed∃",
        "neg": "random",
        "f1": 0.924,
        "xf1": 0.497,
        "xf1_star": 0.193,
    },
    {
        "dataset": "Snomed∃",
        "neg": "hard",
        "f1": 0.872,
        "xf1": 0.460,
        "xf1_star": 0.176,
    },
]

TABLE_1_STATS: list[dict[str, Any]] = [
    {
        "dataset": "FoodOnA",
        "concepts": 30995,
        "removed_axioms": 2244,
        "pall": 102872,
        "splits_2a_2b_3a_3b": "813/777/977/23",
    },
    {
        "dataset": "SnomedA",
        "concepts": 364352,
        "removed_axioms": 18300,
        "pall": 1181331,
        "splits_2a_2b_3a_3b": "651/569/897/103",
    },
    {
        "dataset": "Snomed∃",
        "concepts": 364352,
        "removed_axioms": 2000,
        "pall": 1774,
        "splits_2a_2b_3a_3b": "-/-/1000/0",
    },
]

TABLE_4_ABLATION_FOODONA_FULL: dict[str, Any] = {
    "dataset": "FoodOnA",
    "stages": "2a+2b+3a+3b",
    "random_f1": 0.960,
    "random_xf1": 0.893,
    "random_xf1_star": 0.793,
    "hard_f1": 0.845,
    "hard_xf1": 0.781,
    "hard_xf1_star": 0.684,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_stats": list(TABLE_1_STATS),
        "table_2_neurowl_ont": list(TABLE_2_NEUROWL_ONT),
        "table_4_full_pipeline": dict(TABLE_4_ABLATION_FOODONA_FULL),
    }
