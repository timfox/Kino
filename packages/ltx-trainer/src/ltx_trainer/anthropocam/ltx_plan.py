"""LTX / field-capture integration hooks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anthropocam.config import AnthropoCamConfig, MOBILE_RESOLUTION


def ltx_integration_plan(cfg: AnthropoCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AnthropoCamConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Field still reference grade before LTX I2V (Anthropocene texture sidecar)",
            "Mobile capture → Flask GPU stylize → reference frame for kino-ltx-style presets",
            "Localized style crop to constrain color without semantic erasure",
        ],
        "recommended_resolution": list(MOBILE_RESOLUTION),
        "complements": ["hist2style", "kino-ltx-style.sh"],
        "note": "Expressive Gram-based NST; Hist2Style is photorealistic histogram grading",
    }
