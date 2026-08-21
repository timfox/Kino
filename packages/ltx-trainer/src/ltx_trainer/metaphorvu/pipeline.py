"""MetaphorVU framework card, paper tables, and smoke demos (arXiv:2605.25488)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.metaphorvu.boost import build_demo_graph, metaphor_boost
from ltx_trainer.metaphorvu.config import MetaphorVUConfig
from ltx_trainer.metaphorvu.kg import MetaphorKnowledgeGraph, top_z_references
from ltx_trainer.metaphorvu.taxonomy import (
    METAPHOR_TYPES,
    benchmark_type_stats,
    taxonomy_card,
)


def framework_card(cfg: MetaphorVUConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MetaphorVUConfig()
    return {
        "name": "MetaphorVU",
        "paper": cfg.paper_arxiv,
        "benchmark": "MetaphorVU-Bench",
        "num_videos": cfg.num_videos,
        "kg_nodes": cfg.kg_nodes,
        "kg_edges": cfg.kg_edges,
        "method": "MetaphorBoost (KG mapping augmentation at inference)",
        "taxonomy_types": len(cfg.taxonomy),
        "project": cfg.project_page,
        "dataset_hf": cfg.benchmark_hf,
    }


def _avg_row(**scores: float) -> dict[str, float]:
    return scores


def table_overall_results() -> dict[str, dict[str, float]]:
    """Table 2 — selected models (average column)."""
    return {
        "Human*": _avg_row(average=83.4),
        "GPT-5": _avg_row(average=63.7),
        "GPT-4o": _avg_row(average=56.8),
        "Gemini-3-Pro": _avg_row(average=63.8),
        "Qwen2.5-VL-7B-Instruct": _avg_row(average=33.8),
        "Qwen3-VL-8B-Thinking": _avg_row(average=52.0),
        "Prompt Engineering": _avg_row(average=52.4),
        "Few-shot Example": _avg_row(average=53.6),
        "MetaphorBoost (Gemini-3-Pro)": _avg_row(average=66.1),
        "MetaphorBoost (Qwen2.5-VL-7B)": _avg_row(average=37.9),
        "MetaphorBoost (Qwen3-VL-8B-Thinking)": _avg_row(average=55.9),
    }


def table_gemini3_by_type() -> dict[str, dict[str, float]]:
    """Table 2 excerpt — Gemini-3-Pro baseline vs MetaphorBoost per type."""
    types = [
        "Body L.",
        "Atmosph. L.",
        "Cultural S.",
        "Natural. S.",
        "Causal M.",
        "Analog. M.",
        "Surreal N.",
        "Perform. N.",
    ]
    baseline = [71.2, 74.0, 75.1, 66.9, 49.4, 58.9, 51.1, 48.1]
    boosted = [71.5, 76.3, 77.5, 66.9, 57.2, 59.1, 57.3, 50.8]
    return {
        "Gemini-3-Pro": dict(zip(types, baseline, strict=True)),
        "MetaphorBoost": dict(zip(types, boosted, strict=True)),
    }


def table_error_analysis() -> dict[str, dict[str, float]]:
    """Table 3 — deficiency proportions (%)."""
    return {
        "Gemini-3-Pro": {
            "wrong_recognition": 10.7,
            "missing_mapping": 27.9,
            "superficial_mapping": 33.7,
            "improper_mapping": 27.7,
        },
        "Qwen3-VL-8B-Thinking": {
            "wrong_recognition": 13.5,
            "missing_mapping": 28.1,
            "superficial_mapping": 28.3,
            "improper_mapping": 30.1,
        },
    }


def table_ablation_metaphorboost() -> dict[str, dict[str, float]]:
    """Table 4 — MetaphorBoost ablations (Qwen3-VL-8B-Thinking average)."""
    return {
        "MetaphorBoost": {"average": 55.9},
        "w/o external augmentation": {"average": 53.4},
        "w/o graph-structure augmentation": {"average": 54.3},
        "w/o metaphor-oriented augmentation": {"average": 52.5},
    }


def table_hyperparameter_query() -> dict[str, dict[str, float]]:
    """Table 6 — query strategy / hyperparameter variants."""
    return {
        "MetaphorBoost (default h=2, z=10)": {"average": 55.9},
        "w/o common connection": {"average": 54.8},
        "h=1, z=10": {"average": 54.5},
        "h=2, z=5": {"average": 55.7},
    }


def training_step_demo(cfg: MetaphorVUConfig | None = None) -> dict[str, Any]:
    """Smoke: keyword identification + KG top-z retrieval on demo graph."""
    cfg = cfg or MetaphorVUConfig()
    elements = [
        "tailcoat pigs",
        "banquet",
        "judge attire",
        "cats under table",
        "food scraps",
    ]
    out = metaphor_boost(elements, cfg=cfg)
    graph = build_demo_graph()
    assert out["references"]
    pig_refs = top_z_references(["pig", "tailcoat"], graph, hops=2, z=5)
    return {
        "num_keywords": out["num_keywords"],
        "num_references": out["num_references"],
        "sample_references": out["references"][:5],
        "pig_neighborhood": pig_refs,
    }


def evaluation_demo(cfg: MetaphorVUConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MetaphorVUConfig()
    step = training_step_demo(cfg)
    overall = table_overall_results()
    ablation = table_ablation_metaphorboost()
    errors = table_error_analysis()

    mapping_fail_pct = (
        errors["Gemini-3-Pro"]["missing_mapping"]
        + errors["Gemini-3-Pro"]["superficial_mapping"]
        + errors["Gemini-3-Pro"]["improper_mapping"]
    )

    return {
        **step,
        "taxonomy_count": len(METAPHOR_TYPES),
        "human_avg": overall["Human*"]["average"],
        "gemini3_avg": overall["Gemini-3-Pro"]["average"],
        "metaphorboost_gemini3_avg": overall["MetaphorBoost (Gemini-3-Pro)"]["average"],
        "boost_gain_gemini3": overall["MetaphorBoost (Gemini-3-Pro)"]["average"]
        - overall["Gemini-3-Pro"]["average"],
        "mapping_failure_pct_gemini3": mapping_fail_pct,
        "ablation_external_drop": ablation["MetaphorBoost"]["average"]
        - ablation["w/o external augmentation"]["average"],
        "boost_beats_baseline": overall["MetaphorBoost (Gemini-3-Pro)"]["average"]
        > overall["Gemini-3-Pro"]["average"],
    }
