"""Gopex / LTX integration metadata for DiffTestGen."""

from __future__ import annotations

from typing import Any

from ltx_trainer.difftestgen.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_GITHUB,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    DiffTestGenConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.difftestgen",
        "tools": [
            "difftestgen_knowledge",
            "difftestgen_benchmarks",
            "difftestgen_eval_demo",
            "difftestgen_smoke",
            "difftestgen_ltx_plan",
            "difftestgen_access",
        ],
        "scripts": ["./scripts/kino-difftestgen.sh", "./scripts/gopex-difftestgen.sh"],
        "aiml": "gopex_agent/fixtures/difftestgen.aiml",
        "doc": "documents/DIFFTESTGEN.md",
    }


def framework_card(config: DiffTestGenConfig | None = None) -> dict[str, Any]:
    cfg = config or DiffTestGenConfig()
    return {
        "name": PAPER_SYSTEM,
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
            "github": PAPER_GITHUB,
        },
        "benchmark": BENCHMARK,
        "components": list(COMPONENTS),
        "initial_tests": cfg.initial_tests,
        "top_k_call_paths": cfg.top_k_call_paths,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "GitHub PR + code base → AST change analysis",
            "core": "Access info (call graph + public API) + union coverage feedback to LLM",
            "oracle": "Behavioral difference: error-type / one-sided error / output mismatch",
            "downstream": "Feed differentiating tests to Testora-style regression classifier",
            "notes": [
                "Private changes need public entry paths (Alg. 1)",
                "Union coverage = (cov_old + cov_new) / (chg_old + chg_new)",
                "Loop until 100% union coverage or saturation",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub simulates access extraction + coverage feedback rounds",
            "Wire live sola-st/DiffTestGen harness for full PR evaluation",
        ],
    }
