"""LTX integration notes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.wavtts.config import WavTTSConfig


def ltx_integration_plan(cfg: WavTTSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WavTTSConfig()
    return {
        "package": "ltx_trainer.wavtts",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "native waveform TTS reference for AV fold audio conditioning experiments",
            "flow-matching + patchification patterns shared with LTX trainer utilities",
        ],
        "validation": "./scripts/gopex-wavtts.sh smoke",
    }
