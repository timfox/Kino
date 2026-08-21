"""LTX integration notes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hybrid_aver.config import HybridAverConfig


def ltx_integration_plan(cfg: HybridAverConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HybridAverConfig()
    return {
        "package": "ltx_trainer.hybrid_aver",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "multimodal AV event tagging for surveillance / teleplay QA",
            "cross-attention fusion patterns adjacent to LTX AV training",
        ],
        "validation": "./scripts/gopex-hybrid-aver.sh smoke",
    }
