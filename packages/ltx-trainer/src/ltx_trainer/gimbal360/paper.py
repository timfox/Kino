"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gimbal360.benchmarks import TABLE1_METRICS, benchmarks_bundle, table1_ours
from ltx_trainer.gimbal360.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL


def framework_card() -> dict[str, Any]:
    ours = table1_ours()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "problem": (
            "Euclidean diffusion priors on ERP panoramas fail due to projective variance "
            "(unposed NFoV) and S1 topological severing at azimuth seams."
        ),
        "method": {
            "canonical_space": "Gravity-aligned ERP with horizon on equator (Horizon360)",
            "dal": "Dense flow + soft-argmin rigid 3-DOF warp into canonical space",
            "teg": "Circular VAE padding + Siamese shift-equivariant diffusion loss",
            "fill_conditioning": "ε_θ(z_t ⊕ M ⊕ z_ref, t, τ(y))",
            "inference": f"Euler {30} steps, CFG {30.0}, per-step azimuth shift (Appendix A.2)",
            "base": "Flux.1-fill-dev + LoRA r=64",
        },
        "horizon360": {"size": 20_000, "card": "ltx_trainer.gimbal360.horizon360.dataset_card"},
        "results": {
            "Structured3D_FID": ours["indoor"]["FID"],
            "CVRG_Pano_FID": ours["outdoor"]["FID"],
            "vs_WorldGen_indoor_FID": ours["indoor"]["FID"] < 65.66,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX implements DAL/TEG stubs, mixture pose sampling, and Table 1 anchors. "
            "Production: Flux-fill + Horizon360 from project page."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.gimbal360.mock import evaluation_smoke

    return evaluation_smoke()
