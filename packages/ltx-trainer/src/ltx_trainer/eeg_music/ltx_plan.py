"""LTX / AV fold integration notes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eeg_music.config import EegMusicConfig


def ltx_integration_plan(cfg: EegMusicConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EegMusicConfig()
    return {
        "package": "ltx_trainer.eeg_music",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "semantic music conditioning probes from multimodal agent tools",
            "contrastive alignment patterns shared with CLAP-conditioned AV training",
        ],
        "not_in_fold": "EEG decoding is orthogonal to LTX video latents; keep as research stub",
        "validation": "./scripts/gopex-eeg-music.sh smoke",
    }
