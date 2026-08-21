"""GOPEX integration hooks for Tahoe."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tahoe.benchmarks import PAPER_ANCHORS, TABLE_2_MAIN
from ltx_trainer.tahoe.config import BENCHMARK, PAPER_ARXIV, PAPER_TITLE, PAPER_URL, TahoeConfig


def integration_metadata() -> dict[str, str]:
    return {
        "package": "ltx_trainer.tahoe",
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "benchmark": BENCHMARK,
        "role": "hint-bank Text-to-SQL with logic planning + Snowflake syntax registry",
    }


def gopex_stub_links() -> dict[str, str]:
    return {
        "livebrowsecomp": "Live deep-search benchmarks for SQL workload discovery",
        "cbp_assist": "Hybrid LLM + capability planning for multi-step SQL agents",
        "gopex_hybrid_memory": "FluxMem routing for hint retrieval graphs",
        "nrp": "Neuro-relational programs over embedded relational exports",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "text_to_sql_hint_bank",
        "pipeline": [
            "spider_snow_schema_ingest",
            "error_driven_hint_learning",
            "hint_bank_merge_and_attribution",
            "scope_aware_semantic_retrieval",
            "logic_planning_then_sql_synthesis",
        ],
        "representation": "Hsyn rules + Hsem trigger/strategy layers",
        "dialect": "snowflake",
        "development_examples": TahoeConfig().development_examples,
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Bootstrap Hint Bank on 113 Spider 2.0–Snow supervised examples (y* + exec results).",
        "Run syntax critic loop with Snowflake compiler; distill atomic diffs into Hsyn.",
        "Semantic diffs from execution mismatch; scope tags General / Database / User.",
        "Batch-merge temporary banks; run Strategy Attribution for eval_stats credibility.",
        "At inference: inject full dialect syntax registry + retrieved semantic strategies.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
        "anchors": PAPER_ANCHORS,
        "table_ii_primary": TABLE_2_MAIN[-2:],  # GPT-5.5 vanilla vs Tahoe
    }


def framework_card(cfg: TahoeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TahoeConfig()
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "benchmark": BENCHMARK,
        },
        "method": {
            "hint_types": ["syntax (Hsyn)", "semantic (Hsem trigger/strategy)"],
            "inference": ["scope filter", "semantic retrieval", "logic plan", "SQL synthesis"],
            "learning": ["multi-sample", "syntax then semantic feedback", "batch merge"],
        },
        "config": cfg.__dict__,
        "integration": integration_metadata(),
    }
