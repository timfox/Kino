"""LTX / GOPEX integration notes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.detectzoo.config import DetectZooConfig


def ltx_integration_plan(cfg: DetectZooConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DetectZooConfig()
    return {
        "package": "ltx_trainer.detectzoo",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "benchmark LTX/Gemma captions vs human-written shot prompts (text detectors)",
            "flag synthetic training clips before native merge (image/audio detectors)",
            "AV-fold QA: score renders with unified DetectionResult schema",
        ],
        "validation": "./scripts/gopex-detectzoo.sh smoke",
        "upstream": cfg.github,
    }
