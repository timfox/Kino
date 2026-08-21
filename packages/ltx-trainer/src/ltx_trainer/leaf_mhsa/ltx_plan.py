"""LTX / remote-sensing hooks for vineyard trait pipelines."""

from __future__ import annotations

from typing import Any

from ltx_trainer.leaf_mhsa.config import LeafMHSAConfig


def ltx_integration_plan(cfg: LeafMHSAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LeafMHSAConfig()
    return {
        "doi": cfg.doi,
        "use_cases": [
            "Synthetic leaf spectra for RTM canopy forward models when traits known",
            "Vineyard monitoring: trait imputation → spectra → canopy reflectance bridge",
            "Conditioning HDR / multispectral LTX plates on grapevine biochemical sidecars",
        ],
        "note": "Species-specific alternative to PROSPECT-PRO; CPU stub only in Gopex",
    }
