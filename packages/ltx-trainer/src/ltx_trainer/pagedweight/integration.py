"""Gopex / LTX integration metadata for PagedWeight."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pagedweight.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    PagedWeightConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.pagedweight",
        "tools": [
            "pagedweight_knowledge",
            "pagedweight_benchmarks",
            "pagedweight_eval_demo",
            "pagedweight_smoke",
            "pagedweight_ltx_plan",
            "pagedweight_plan_demo",
        ],
        "scripts": ["./scripts/kino-pagedweight.sh", "./scripts/gopex-pagedweight.sh"],
        "aiml": "gopex_agent/fixtures/pagedweight.aiml",
        "doc": "documents/PAGEDWEIGHT.md",
    }


def framework_card(config: PagedWeightConfig | None = None) -> dict[str, Any]:
    cfg = config or PagedWeightConfig()
    return {
        "name": PAPER_SYSTEM,
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "benchmark": BENCHMARK,
        "components": list(COMPONENTS),
        "n_experts": cfg.n_experts,
        "n_layers": cfg.n_layers,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "MoE Any-Precision bit-plane + LUT weight pages on vLLM",
            "core": "Quality-aware planner: offline Hessian sensitivity + routing buckets + prompt residual",
            "movement": "Async GPU↔CPU offload/reload at safe commit boundaries",
            "kernel": "Fused mixed-precision MoE AP kernel (per gate_up/down bitwidth)",
            "notes": [
                "Complements PagedAttention: pages MoE weights under KV pressure",
                "FP16-equiv quality at up to 72% GPU mem savings / 1.94× throughput",
                "≤4.1% TPS loss vs uniform APL baseline",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub simulates page table + planner + async offload",
            "Wire real APL MoE kernel + vLLM KV free-block signals for production",
        ],
    }
