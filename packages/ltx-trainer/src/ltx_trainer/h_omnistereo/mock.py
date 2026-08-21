"""Runnable evaluation smoke for H-OmniStereo (arXiv:2605.14963)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.h_omnistereo.benchmarks import TABLE2_STEREO, benchmarks_bundle
from ltx_trainer.h_omnistereo.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "h_omnistereo",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_3d60_mae": TABLE2_STEREO["Ours"]["3d60_mae"],
        "ref_mvs_gi_mae": TABLE2_STEREO["Ours"]["mvs_gi_mae"],
    }
    try:
        import torch

        from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
        from ltx_trainer.h_omnistereo.heading_normal import camera_to_heading_aligned, longitude_grid
        from ltx_trainer.h_omnistereo.pipeline import evaluation_demo_run
        from ltx_trainer.h_omnistereo.spherical_geo import spherical_disparity

        cfg = HOmniStereoConfig(train_crop_w=48, train_crop_h=48, feature_dim=24, cost_groups=4, refine_iters=2)
        demo = evaluation_demo_run(cfg, device="cpu", refine_iters=2)
        alpha = longitude_grid(48, 48)
        n = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1).expand(1, 3, 48, 48)
        n_ha = camera_to_heading_aligned(n, alpha.unsqueeze(0))
        disp = spherical_disparity(
            torch.tensor(0.3),
            torch.tensor(0.35),
            torch.tensor(2.0),
            0.2,
        )

        out.update(
            {
                "torch": True,
                "heading_aligned_normal_finite": bool(torch.isfinite(n_ha).all()),
                "spherical_disparity": float(disp.item()),
                "demo_normal_mae_deg": demo["normal"]["normal_mae_deg"],
                "demo_stereo_mae": demo["stereo"]["mae"],
                "demo_uncertainty_loss": demo["uncertainty"]["loss_conf"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
