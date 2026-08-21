"""PhysHDR-GS paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.physthdr_gs.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle
from ltx_trainer.physthdr_gs.config import PROJECT_URL


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    t1 = b["table1_exp3_ours_dagger"]
    t2 = b["table2_exp3_syn"]
    t3 = b["table3_efficiency"]
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "project": PROJECT_URL,
        "authors": "Zeng, Bai, Wang, Fu (Northeastern University)",
        "problem": (
            "HDR-NVS from multi-exposure LDR views entangles material and illumination; "
            "tone-mapped LDR supervision starves gradients in under/over-exposed regions."
        ),
        "method": {
            "ie_branch": "Exposure t scales rasterized HDR (image-exposure)",
            "gi_branch": "Virtual ambient illumination φ(La, l) relights Gaussians",
            "losses": "Lrec (dual-branch LDR) + Lcons (cross-branch HDR) + optional Lunit",
            "igs": "Illumination-guided gradient scaling for densification",
        },
        "results": {
            "nerf_real_ldr_oe_psnr": t1["hdr_nerf_real_ldr_oe"]["psnr"],
            "syn_hdr_psnr_ours_dagger": t2["ours_dagger"]["hdr"]["psnr"],
            "syn_hdr_psnr_hdr_gs": t2["hdr_gs"]["hdr"]["psnr"],
            "render_fps_ours": t3["ours"]["fps"],
            "psnr_gain_over_hdr_gs_db": b["hdr_psnr_gain_over_hdr_gs_db"],
        },
        "reference_metrics": b,
        "integration": (
            "Trainable PyTorch PhysHDR-GS (perspective splat, IE/GI, I-GS densify); "
            "reference Table 1–4 numbers; no official upstream weights."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.physthdr_gs.mock import evaluation_smoke

    return evaluation_smoke()
