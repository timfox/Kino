"""Paper card export."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gsparc.benchmarks import benchmarks_bundle
from ltx_trainer.gsparc.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.gsparc.datasets import datasets_card
from ltx_trainer.gsparc.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "authors": [
            "Bhavya Sai Nukapotula",
            "Rishabh Tripathi",
            "Seth Pregler",
            "Dileep Kalathil",
            "Srinivas Shakkottai",
            "Tedd Rappaport",
        ],
        "affiliation": "Texas A&M University / NYU",
        "components": [
            "Anisotropic 3D Gaussians + per-Gaussian emission MLP (3→16→2)",
            "Physics-informed log-distance path loss (Eq. 1)",
            "Transmittance-based RF compositing (Eq. 3; 1/d ablation Sec. 7.1)",
            "Hemispherical equirectangular splatting (CUDA in paper; CPU stub here)",
            "DUSt3R-style confidence-weighted loss (Eq. 6)",
            "Adaptive densification / prune (Algorithm 1 stub)",
            "Spectrum (L_sp) or complex CSI (L_ch) objectives",
        ],
        "contributions": [
            "Low-ms inference vs NeRF2/GSRF (0.8–3.8 ms render)",
            "Per-position confidence → ~71% pilot-free @ τ=0.59 (Argos)",
            "Sionna + RFID + Argos evaluation; Sionna PHY downstream",
        ],
        "datasets": datasets_card(),
        "benchmarks": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "gsparc", **evaluation_demo_run()}
