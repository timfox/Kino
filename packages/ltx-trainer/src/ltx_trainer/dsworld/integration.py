"""Gopex / LTX integration metadata for DSWorld."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsworld.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_GITHUB,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    DSWorldConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.dsworld",
        "tools": [
            "dsworld_knowledge",
            "dsworld_benchmarks",
            "dsworld_eval_demo",
            "dsworld_smoke",
            "dsworld_ltx_plan",
            "dsworld_route",
        ],
        "scripts": ["./scripts/kino-dsworld.sh", "./scripts/gopex-dsworld.sh"],
        "aiml": "gopex_agent/fixtures/dsworld.aiml",
        "doc": "documents/DSWORLD.md",
    }


def framework_card(config: DSWorldConfig | None = None) -> dict[str, Any]:
    cfg = config or DSWorldConfig()
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
        "backbone": cfg.backbone,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "State Constructor SC(E) → structured {T, D, P, L}",
            "core": "St+1 = W(St, At) via Router → Compiler | LLM Simulator",
            "train": "SFT warm-up + Reflective World Model Optimization (GRPO)",
            "data": "DSWorld-8K real + synthetic verified transitions with CoT",
            "notes": [
                "Cost-aware: light ops execute, heavy ops simulate",
                "~14× RL training / 3-6× search inference speedup",
                "+35.6% avg over o4-mini on transition prediction",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub simulates routing + hybrid transitions + reflective RL",
            "Wire live DSWorld harness for MLE-Bench / agent training",
        ],
    }
