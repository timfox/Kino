"""LTX integration plan for VISA evidence sidecars."""

from __future__ import annotations

from typing import Any

from ltx_trainer.visa.config import VisaConfig


def ltx_integration_plan(cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_case": "MMAR-style audio QA evidence in AV-fold sidecars",
        "preprocess": [
            "multimodal_feature_bundle → visa.features JSON in conditions/",
            "agentic_sed timestamps → fold sidecar for temporal QA",
        ],
        "inference": [
            "Optional: route LTX audio-ref IC masks using SED event spans",
            "VLM spectrogram views as conditioning images for clip-level QA",
        ],
        "training": "No DiT gradient — agent orchestration only",
    }
