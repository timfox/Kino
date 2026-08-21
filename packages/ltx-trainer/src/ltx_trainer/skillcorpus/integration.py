"""Gopex / LTX integration metadata for SkillCorpus."""

from __future__ import annotations

from typing import Any

from ltx_trainer.skillcorpus.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    SkillCorpusConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.skillcorpus",
        "tools": [
            "skillcorpus_knowledge",
            "skillcorpus_benchmarks",
            "skillcorpus_eval_demo",
            "skillcorpus_smoke",
            "skillcorpus_ltx_plan",
            "skillcorpus_retrieve",
        ],
        "scripts": ["./scripts/kino-skillcorpus.sh", "./scripts/gopex-skillcorpus.sh"],
        "aiml": "gopex_agent/fixtures/skillcorpus.aiml",
        "doc": "documents/SKILLCORPUS.md",
    }


def framework_card(config: SkillCorpusConfig | None = None) -> dict[str, Any]:
    cfg = config or SkillCorpusConfig()
    return {
        "name": PAPER_SYSTEM,
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "benchmark": BENCHMARK,
        "components": list(COMPONENTS),
        "n_active": cfg.n_active_skills,
        "n_raw": cfg.n_raw_crawl,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "62-source crawl → 6-stage funnel (parse, dedup, 3-facet judge, safety+licence)",
            "core": "96,401 OSI-permissive skills; 16-class taxonomy; utility/robustness/safety",
            "match": "Fine-tuned Emb+Rank 0.6B → LLM selector (0–2 full-body skills)",
            "eval": "SkillsBench / GDPVal / QwenClawBench × OpenClaw / Raven",
            "notes": [
                "Pooled SkillsBench +7.5 pp; Raven×Q-397B +13.4; Opus +8.0",
                "Gains track coverage (retrieval-match) and harness execute–verify loop",
                "Ablation: curation and fine-tuned retrieval each contribute",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub: quality facets, hard gates, retrieve-select, funnel anchors",
            "Wire live SkillCorpus index for Gopex Claw / OpenClaw skill loading",
        ],
    }
