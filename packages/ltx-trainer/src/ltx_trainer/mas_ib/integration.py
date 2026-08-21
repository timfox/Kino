"""Gopex / LTX integration metadata for MAS-IB."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mas_ib.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_GITHUB,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    MasIbConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.mas_ib",
        "tools": [
            "mas_ib_knowledge",
            "mas_ib_benchmarks",
            "mas_ib_eval_demo",
            "mas_ib_smoke",
            "mas_ib_ltx_plan",
            "mas_ib_gain",
        ],
        "scripts": ["./scripts/kino-mas-ib.sh", "./scripts/gopex-mas-ib.sh"],
        "aiml": "gopex_agent/fixtures/mas_ib.aiml",
        "doc": "documents/MAS_IB.md",
    }


def framework_card(config: MasIbConfig | None = None) -> dict[str, Any]:
    cfg = config or MasIbConfig()
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
        "beta": cfg.beta,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "Agentic task + planner-induced subtask chain",
            "core": "Relay IB: min I(M;m) − β I(m;Y|X); G = H(M|m) − β Δ(m)",
            "prototypes": "SAS vs SAS-contextflow vs MAS under matched step budgets",
            "design": "Optimize relays to drop upstream noise while preserving downstream-relevant bits",
            "notes": [
                "Unbounded relays → MAS ≡ SAS (Prop. 3.1)",
                "MAS helps when context reduction outweighs capability-weighted relay loss",
                "Stronger models (↑β) gain less from compression / lose more from Δ>0",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub simulates IB gain + prototype comparison",
            "Wire live ALFWorld/WebShop harness from divelab/MAS-SAS for full eval",
        ],
    }
