"""Evaluation smoke (arXiv:2602.09076)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.legs_over_arms.benchmarks import TABLE1_JRDB_3D, benchmarks_bundle
from ltx_trainer.legs_over_arms.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    best = next(r for r in TABLE1_JRDB_3D if r["G"] == "K3D_L")
    out: dict[str, Any] = {
        "package": "legs_over_arms",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_minADE_K3D_L": best["minADE"],
    }
    try:
        import torch

        from ltx_trainer.legs_over_arms.biomechanics import cues_lower_body_3d
        from ltx_trainer.legs_over_arms.config import LegsOverArmsConfig
        from ltx_trainer.legs_over_arms.erp_tracks import erp_pixel_to_robot_xy, tracks_from_keypoints_stub
        from ltx_trainer.legs_over_arms.hst_net import HSTStub
        from ltx_trainer.legs_over_arms.pipeline import evaluation_demo_run
        from ltx_trainer.legs_over_arms.skeleton import total_feature_dim

        cfg = LegsOverArmsConfig()
        demo = evaluation_demo_run(cfg, device="cpu", feature_config="K3D_L")
        past = tracks_from_keypoints_stub(num_agents=cfg.num_agents, steps=cfg.history_steps)
        model = HSTStub(cfg, feature_config="K3D_L")
        pose = torch.randn(1, cfg.num_agents, total_feature_dim("K3D_L"))
        o = model(past, pose)
        kp = torch.randn(1, 6, 10, 3)
        cues = cues_lower_body_3d(kp)
        xy = erp_pixel_to_robot_xy(torch.tensor([64.0]), torch.tensor([32.0]))
        out.update(
            {
                "torch": True,
                "demo_loss": demo["train"]["loss"],
                "pred_modes_shape": list(o["pred_modes"].shape),
                "cues_shape": list(cues.shape),
                "erp_xy": xy.tolist(),
                "ade_reduction_pct": benchmarks_bundle()["key_finding"][
                    "K3D_L_minADE_reduction_pct"
                ],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
