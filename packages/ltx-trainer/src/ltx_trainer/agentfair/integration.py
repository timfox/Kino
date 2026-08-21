"""Gopex / LTX integration metadata for AgentFAIR."""

from __future__ import annotations

from typing import Any

from ltx_trainer.agentfair.config import (
    BENCHMARK,
    AgentFAIRConfig,
    PAPER_ARXIV,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_ZENODO,
)


def framework_card(config: AgentFAIRConfig | None = None) -> dict[str, Any]:
    cfg = config or AgentFAIRConfig()
    return {
        "name": "AgentFAIR",
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
            "zenodo": PAPER_ZENODO,
        },
        "benchmark": BENCHMARK,
        "subprinciples": 13,
        "maturity_scale": "0-3",
        "critic_enabled_default": cfg.enable_critic,
        "model_default": cfg.model_name,
        "mean_cost_usd": cfg.mean_cost_usd,
        "geo_indicators": list(cfg.geo_indicators),
        "stages": [
            "Playwright crawl + hybrid metadata extraction",
            "13 parallel sub-principle evaluators (F/A/I/R)",
            "Critic evidence/consistency checks + retry",
            "Markdown/JSON/SQLite audit artifacts",
        ],
        "integration": integration_metadata(),
    }


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.agentfair",
        "tools": [
            "agentfair_knowledge",
            "agentfair_benchmarks",
            "agentfair_eval_demo",
            "agentfair_smoke",
            "agentfair_ltx_plan",
            "agentfair_evaluate",
        ],
        "scripts": [
            "./scripts/kino-agentfair.sh",
            "./scripts/gopex-agentfair.sh",
        ],
        "aiml": "gopex_agent/fixtures/agentfair.aiml",
        "doc": "documents/AGENTFAIR.md",
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "Landing-page URL list (DOI/Handle) + optional repository hints",
            "extract": "Playwright JS render + Schema.org/JSON-LD/DCAT hybrid parse",
            "evaluate": "13 rubric agents + deterministic G checks",
            "critic": "Evidence sufficiency + cross-principle consistency retries",
            "export": "Markdown report, JSON assessment, SQLite evidence store",
            "notes": [
                "FAIRness ≠ algorithmic fairness",
                "Normalized baseline scores are disagreement diagnostics only",
                "I2 registry criterion can under-credit community geospatial vocabularies",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub uses rule-based agents (no OpenAI calls)",
            "Wire GPT-4o-mini via env for live AgentFAIR when needed",
            "Prefer Colibri / local OpenAI base for on-prem runs",
        ],
    }
