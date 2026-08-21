"""Paper card and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dynamic_gp.basis import morphology_card
from ltx_trainer.dynamic_gp.benchmarks import heat_error_curve, summary_anchors, wave_example_card
from ltx_trainer.dynamic_gp.constants import (
    ERROR_DECOMPOSITION,
    PAPER_ARXIV,
    PAPER_CODE,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TRAIN_DEFAULTS,
)
from ltx_trainer.dynamic_gp.error_analysis import error_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "venue": PAPER_VENUE,
        "code": PAPER_CODE,
        "authors": "J.S. van Hulst, W.P.M.H. Heemels, D.J. Antunes",
        "affiliation": "Eindhoven University of Technology",
        "findings": [
            "DGP: GP posterior closed under IDE evolution + observations (Thm. 3.1–3.2)",
            "Separable kernels → finite-dimensional Kalman filter on z_t ∈ R^{DM}",
            "L2 error decomposes: noise-limited + leakage gap + out-of-subspace (Eq. 5.14)",
            "Heat (D=1) and wave (D=2) PDE examples with Fourier basis",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "separable_kernels": morphology_card(),
        "error_analysis": error_card(),
        "error_terms": list(ERROR_DECOMPOSITION),
        "train_defaults": dict(TRAIN_DEFAULTS),
        "heat_error_curve": heat_error_curve(),
        "wave_example": wave_example_card(),
        "summary": summary_anchors(),
    }
