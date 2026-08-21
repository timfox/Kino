"""Framework card and benchmarks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.augment_engineering.config import AugmentEngConfig
from ltx_trainer.augment_engineering.layout import LIMITATIONS, PIPELINE_STAGES
from ltx_trainer.augment_engineering.mock import evaluation_smoke
from ltx_trainer.augment_engineering.phases import orchestration_patterns, phase_catalog
from ltx_trainer.augment_engineering.rubric import rubric_levels
from ltx_trainer.augment_engineering.tables import (
    headline_results,
    longitudinal_phases,
    table3_stack_inventory,
    table6_pipeline_overhead,
)


def framework_card(cfg: AugmentEngConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AugmentEngConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Organizations deploy many purpose-built AI tools per domain; augment engineering "
            "orchestrates them using portable prompt and context engineering skills."
        ),
        "research_questions": [
            "RQ1: Are PE/CE skills portable across tools and domains?",
            "RQ2: Can multi-tool orchestration be codified systematically?",
        ],
        "three_discipline_progression": [
            "Prompt Engineering (one tool, one task)",
            "Context Engineering (reproducible pipelines, one tool)",
            "Augment Engineering (portfolio across domains)",
        ],
        "six_phases": [p["name"] for p in phase_catalog()],
        "portability_metrics": list(cfg.portability_metrics),
        "orchestration_patterns": [p["name"] for p in orchestration_patterns()],
        "case_study": {
            "period": "Nov 2025 – Mar 2026",
            "domains": list(cfg.case_study_domains),
            "ai_tools": list(cfg.ai_tools),
            "infrastructure": list(cfg.infrastructure),
        },
        "prompt_rubric": [f"{r.level}: {r.name}" for r in rubric_levels()],
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "phase_catalog": phase_catalog(),
        "orchestration_patterns": orchestration_patterns(),
        "table3_stack": table3_stack_inventory(),
        "table6_pipeline_overhead": table6_pipeline_overhead(),
        "longitudinal_phases": longitudinal_phases(),
        "rubric_levels": [{"level": r.level, "name": r.name} for r in rubric_levels()],
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
