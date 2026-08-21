"""LightHarmony3D paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lightharmony3d.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t1 = b["table1_lh3d_ku"]["lightharmony3d"]
    t2 = b["table2_vqa_mipnerf360"]["lightharmony3d"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Huang, Ren, Wan, Zheng, Wang, Chen, Gong, Liu (USyd / Melbourne / Google)",
        "problem": (
            "Inserting explicit meshes into 3DGS scenes needs consistent HDR lighting and cast shadows; "
            "inverse rendering is costly and per-view HDR methods break multi-view coherence."
        ),
        "method": {
            "reconstruction": "MILo hybrid Gaussian–mesh",
            "gen_env": "GenEnvLighting: Flux Kontext LoRA EV0→EV−3, exposure fusion to HDR",
            "visibility": "Ray-decoupled BSDF for enclosed scenes",
            "composite": "PBR shadow-ratio modulation over 3DGS background",
        },
        "results": {
            "lh3d_ku_psnr": t1["psnr"],
            "lh3d_ku_lpips": t1["lpips"],
            "vqa_ratio": t2["ratio"],
        },
        "reference_metrics": b,
        "integration": "Complements 3DGS editing stacks; stub only (no Flux/MILo weights).",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.lightharmony3d.mock import evaluation_smoke

    return evaluation_smoke()
