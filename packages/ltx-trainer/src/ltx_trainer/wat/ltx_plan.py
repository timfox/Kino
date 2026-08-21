"""LTX integration notes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.wat.config import WATConfig


def ltx_integration_plan(cfg: WATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WATConfig()
    return {
        "package": "ltx_trainer.wat",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "unified audio/image/video latent budgeting for AV fold preprocessing",
            "energy-based sparse token selection prior to LTX conditioning",
        ],
        "validation": "./scripts/gopex-wat.sh smoke",
    }
