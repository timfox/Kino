"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphere_depth.benchmarks import TABLE2_DEPTH_ERRORS, benchmarks_bundle
from ltx_trainer.sphere_depth.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    acd = next(r for r in TABLE2_DEPTH_ERRORS if r["model"] == "ACDNet")
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "problem": (
            "360° depth models assume gravity-aligned ERP (canonical pose). "
            "Real robotic mounts introduce pitch/roll that break depth accuracy."
        ),
        "method": {
            "benchmark": "8 indoor ERP images, 37 metric landmarks, pose splits",
            "calibration": "Learn λ on train landmarks; MSE ε on test landmarks",
            "pose_study": "Simulated pitch/roll grids (±40° coarse, ±2° fine)",
            "models": "ACDNet, DepthAnywhere, BiFuse++, SliceNet, Depth Anything v2",
        },
        "results": {
            "ACDNet_gravity_aligned_m": acd["gravity_aligned"],
            "ACDNet_high_deformation_m": acd["high_def"],
            "DepthAnywhere_lambda": 0.56,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX implements λ calibration, ERP pose rotation, cubemap stub, and Table 2 anchors. "
            "Production: github.com/sgazzeh/Sphere_depth."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.sphere_depth.mock import evaluation_smoke

    return evaluation_smoke()
