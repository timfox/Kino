"""Runnable evaluation smoke for Sphere-Depth (arXiv:2604.23432)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphere_depth.benchmarks import TABLE2_DEPTH_ERRORS, benchmarks_bundle
from ltx_trainer.sphere_depth.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    acd = next(r for r in TABLE2_DEPTH_ERRORS if r["model"] == "ACDNet")
    out: dict[str, Any] = {
        "package": "sphere_depth",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_acdnet_gravity_m": acd["gravity_aligned"],
        "ref_acdnet_high_def_m": acd["high_def"],
    }
    try:
        import torch

        from ltx_trainer.sphere_depth.calibration import learn_scaling_lambda
        from ltx_trainer.sphere_depth.config import SphereDepthConfig
        from ltx_trainer.sphere_depth.disparity import disparity_to_depth
        from ltx_trainer.sphere_depth.pipeline import evaluation_demo_run
        from ltx_trainer.sphere_depth.pose import apply_pose_perturbation

        cfg = SphereDepthConfig(erp_height=48, erp_width=96)
        demo = evaluation_demo_run(cfg, device="cpu")
        erp = torch.rand(1, 3, 48, 96)
        warped = apply_pose_perturbation(erp, pitch_deg=10.0, roll_deg=-5.0)
        disp_depth = disparity_to_depth(torch.zeros(1, 48, 96))
        lam = learn_scaling_lambda(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.1, 2.0, 2.9]))
        out.update(
            {
                "torch": True,
                "warped_finite": bool(torch.isfinite(warped).all()),
                "disp_depth_range": [float(disp_depth.min()), float(disp_depth.max())],
                "lambda_smoke": lam,
                "demo_acdnet_eps": demo["ACDNet"]["epsilon_test"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
