"""Gopex / LTX integration metadata for Yi."""

from __future__ import annotations

from typing import Any

from ltx_trainer.yi.config import (
    BENCHMARK,
    COMPONENTS,
    PAPER_ARXIV,
    PAPER_SYSTEM,
    PAPER_TITLE,
    PAPER_URL,
    YiConfig,
)


def integration_metadata() -> dict[str, Any]:
    return {
        "package": "ltx_trainer.yi",
        "tools": [
            "yi_knowledge",
            "yi_benchmarks",
            "yi_eval_demo",
            "yi_smoke",
            "yi_ltx_plan",
            "yi_update_demo",
        ],
        "scripts": ["./scripts/kino-yi.sh", "./scripts/gopex-yi.sh"],
        "aiml": "gopex_agent/fixtures/yi.aiml",
        "doc": "documents/YI.md",
    }


def framework_card(config: YiConfig | None = None) -> dict[str, Any]:
    cfg = config or YiConfig()
    return {
        "name": PAPER_SYSTEM,
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "benchmark": BENCHMARK,
        "components": list(COMPONENTS),
        "principle": "decomposition facilitates consolidation",
        "max_out_degree": cfg.max_out_degree,
        "buffer_gb": cfg.buffer_gb,
        "worker_threads": cfg.worker_threads,
        "integration": integration_metadata(),
    }


def integration_bundle() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "ltx_plan": {
            "ingest": "On-disk proximity graph + streaming insert/delete stream",
            "core": "Vector-level connect tasks unify Expand+Prune for insert/delete",
            "engine": "C++20 coroutine tasklets (search/connect) overlapping I/O",
            "buffer": "Async fixed-budget page cache with dirty/free lists",
            "vfs": "Split index topology vs raw vector data blocks",
            "notes": [
                "In-place insert AND delete (unlike OdinANN / DiskANN / IP-DiskANN alone)",
                "Avoids offline merge cost that grows with index size",
                "Fewer threads than OdinANN while higher concurrent search QPS",
            ],
        },
        "gopex_stubs": integration_metadata(),
        "prep_notes": [
            "CPU stub simulates connect/delete LRU + tasklet/buffer/VFS",
            "Wire real DiskANN-style graph + SPDK for production Yi",
        ],
    }
