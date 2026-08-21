"""LTX / AV fold hooks for unified audio-text diffusion."""

from __future__ import annotations

from typing import Any

from ltx_trainer.uat.config import UATConfig


def ltx_integration_plan(cfg: UATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UATConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "TTA conditioning sidecar for LTX AV fold (text prompt → latent audio prior)",
            "Audio caption export from generated clips for catalog / audit",
            "SDEdit-style latent edit bridge between source clip and new prompt",
        ],
        "note": "Diffusion-centric unified model; CPU stub only in Gopex (no AudioX weights)",
    }
