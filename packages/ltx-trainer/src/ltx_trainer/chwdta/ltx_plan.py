"""LTX hooks: learned compression priors for video latents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.chwdta.config import ChwdtaConfig


def ltx_integration_plan(cfg: ChwdtaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChwdtaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "HDR/SDR frame latent ChWP slice ordering for entropy coding sidecars",
            "ChWDTB-style channel covariance sparsification in transformer blocks",
            "Rate–complexity trade-off: 4-slice vs 8-slice entropy for delivery tiers",
        ],
        "note": "Image LIC codec; video extension would stack ChWP per-frame latents",
    }
