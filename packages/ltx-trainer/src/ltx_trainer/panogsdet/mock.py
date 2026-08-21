"""Runnable evaluation smoke for PanoGSDet (arXiv:2605.14601)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panogsdet.benchmarks import TABLE1_AP25, benchmarks_bundle
from ltx_trainer.panogsdet.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "panogsdet",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_mAP25": TABLE1_AP25["mAP@25"],
        "ref_mAP50": 0.2461,
    }
    try:
        import torch

        from ltx_trainer.panogsdet.config import PanoGSDetConfig
        from ltx_trainer.panogsdet.erp_geometry import depth_map_to_points
        from ltx_trainer.panogsdet.pipeline import evaluation_demo_run

        cfg = PanoGSDetConfig(height=48, width=96, feature_dim=16, opt_blocks=1)
        demo = evaluation_demo_run(cfg, device="cpu", face_size=16, max_gaussians=128)
        depth = torch.rand(1, 1, 48, 96)
        pts = depth_map_to_points(depth)
        out.update(
            {
                "torch": True,
                "points_finite": bool(torch.isfinite(pts).all()),
                "demo_loss": demo["train"]["loss"],
                "demo_proposals": demo["proposal_count"],
                "gaussian_count": demo["num_gaussians"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
