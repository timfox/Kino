"""Paper anchors and table stubs for SkillCorpus (arXiv:2607.15557)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "SkillCorpus",
    "n_raw_crawl": 821000,
    "n_active": 96401,
    "n_taxonomy_classes": 16,
    "n_quality_facets": 3,
    "skillsbench_pooled_delta_pp": 7.5,
    "gdpval_pooled_delta_pp": 1.51,
    "qwenclaw_pooled_delta_pp": 2.79,
    "strongest_cell_delta_pp": 13.4,  # Raven × Q-397B SkillsBench
    "opus_delta_pp": 8.0,
    "ablation_full_pass": 22.6,
    "ablation_no_skill": 9.2,
    "hit_at_1_skillcorpus": 0.720,
    "recall_at_10_skillcorpus": 0.718,
}

# Table 1 absolute % (SkillsBench / GDPVal×100 / QwenClaw×100), no-skill → skill
TABLE_1_CELLS: list[dict[str, Any]] = [
    {
        "cell": "OpenClaw × Q-27B",
        "skillsbench": (8.8, 13.0),
        "gdpval": (81.2, 83.1),
        "qwenclaw": (65.2, 66.7),
    },
    {
        "cell": "OpenClaw × Q-397B",
        "skillsbench": (11.1, 16.9),
        "gdpval": (82.2, 84.0),
        "qwenclaw": (65.7, 67.0),
    },
    {
        "cell": "Raven × Q-27B",
        "skillsbench": (10.0, 16.5),
        "gdpval": (82.6, 83.8),
        "qwenclaw": (66.9, 70.8),
    },
    {
        "cell": "Raven × Q-397B",
        "skillsbench": (9.2, 22.6),
        "gdpval": (84.0, 85.2),
        "qwenclaw": (68.8, 73.2),
    },
]

TABLE_1_POOLED_DELTA: dict[str, float] = {
    "SkillsBench": 7.5,
    "GDPVal": 1.51,
    "QwenClawBench": 2.79,
}

# Table 2 ablation (Raven × Q-397B, SkillsBench)
TABLE_2_ABLATION: list[dict[str, Any]] = [
    {"corpus": "—", "retrieval": "—", "pass": 9.2, "delta": None},
    {"corpus": "Filtered", "retrieval": "Fine-tuned (ours)", "pass": 22.6, "delta": 13.4},
    {"corpus": "Filtered", "retrieval": "Off-the-shelf Qwen3", "pass": 13.8, "delta": 4.6},
    {"corpus": "Raw crawl", "retrieval": "Fine-tuned (ours)", "pass": 14.9, "delta": 5.7},
]

# Table 7 standalone retrieval (SkillCorpus pool)
TABLE_7_RETRIEVAL: list[dict[str, Any]] = [
    {"method": "Base", "hit1": 0.400, "r10": 0.635},
    {"method": "SkillRouter", "hit1": 0.613, "r10": 0.691},
    {"method": "Ours", "hit1": 0.720, "r10": 0.718},
]

# Top taxonomy shares (Fig. 2)
TAXONOMY_SHARES: dict[str, float] = {
    "Dev": 22.4,
    "Data": 14.1,
    "Writing": 8.2,
    "DevOps-Infra": 7.8,
    "Multimedia": 7.5,
    "Testing": 6.4,
    "AI-ML": 5.5,
    "Other": 2.0,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_cells": list(TABLE_1_CELLS),
        "table_1_pooled_delta": dict(TABLE_1_POOLED_DELTA),
        "table_2_ablation": list(TABLE_2_ABLATION),
        "table_7_retrieval": list(TABLE_7_RETRIEVAL),
        "taxonomy_shares": dict(TAXONOMY_SHARES),
    }
