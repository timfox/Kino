"""LTX / spatial audio hooks for GCP-ISM RIRs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gcp_ism.config import GCPIsmConfig


def ltx_integration_plan(cfg: GCPIsmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GCPIsmConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "High-N orthotope RIR priors for synthetic reverb sidecars on teleplay shots",
            "Compare GCP-ISM tail density vs FDN/hybrid RIR in energy-aware eval",
            "Integer-room probe for modal density vs classic 3D ISM in agent tools",
        ],
        "note": "CPU stub; full GCP-ISM LUT+FFT path in github.com/yluo1/GCP-ISM",
    }
