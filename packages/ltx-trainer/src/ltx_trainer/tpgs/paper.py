"""Framework card for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tpgs.config import GITHUB_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "github": GITHUB_URL,
        "method": "TPGS",
        "latent_space": "cubemap + transition planes (perspective 3DGS)",
        "stages": ["intra-face cube+TP views", "inter-face stitched ERP"],
        "components": [
            "Transition-plane cubemap rasterization with 12 views (6 faces + 6 TP)",
            "Cube padding via spherical sampling (Eq. 8–9)",
            "Intra L1+D-SSIM on perspective views (Eq. 10)",
            "Inter ERP consistency after stitch (Eq. 11–12)",
        ],
        "vs_odgs": "perspective 3DGS backbone; ~24GB VRAM vs ODGS ~48GB (paper)",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.tpgs.pipeline import evaluation_demo_run

    return {"package": "tpgs", **evaluation_demo_run()}
