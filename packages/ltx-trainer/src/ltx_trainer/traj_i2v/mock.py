"""Runnable evaluation smoke for trajectory-guided I2V (arXiv:2605.16420)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.traj_i2v.benchmarks import TABLE3_QUANTITATIVE, benchmarks_bundle
from ltx_trainer.traj_i2v.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "traj_i2v",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_sg_i2v_traj_px": TABLE3_QUANTITATIVE["sg_i2v"]["trajectory_error_px"],
        "ref_gt_brisque": TABLE3_QUANTITATIVE["ground_truth"]["brisque"],
    }
    try:
        import torch

        from ltx_trainer.traj_i2v.conditioning import build_sg_i2v_conditioning
        from ltx_trainer.traj_i2v.gps_mapping import estimate_scale_px_per_m
        from ltx_trainer.traj_i2v.pipeline import reconstruct_clip, sg_i2v_generate_stub
        from ltx_trainer.traj_i2v.synthetic import (
            synthetic_anchors,
            synthetic_gps_log,
            synthetic_reference_frame,
        )

        cfg = __import__(
            "ltx_trainer.traj_i2v.config", fromlist=["TrajI2VConfig"]
        ).TrajI2VConfig(image_width=96, image_height=54, num_frames=6)
        ref = synthetic_reference_frame(cfg)
        log = synthetic_gps_log(cfg)
        anchors = synthetic_anchors(cfg)
        scale = estimate_scale_px_per_m(anchors[0], anchors[1], anchors[0].lon0, anchors[0].lat0)
        cond = build_sg_i2v_conditioning(ref, log, anchors, cfg=cfg, scale_px_per_m=scale)
        frames = sg_i2v_generate_stub(cond, cfg=cfg)
        result = reconstruct_clip(ref, log, anchors, cfg=cfg, method="optical_flow")

        out.update(
            {
                "torch": True,
                "scale_px_per_m": round(scale, 3),
                "num_objects": len(cond.objects),
                "generated_frames": int(frames.shape[0]),
                "optical_flow_temporal": round(result["metrics"]["temporal_smoothness"], 4),
                "six_object_conditioning": len(cond.objects) == 6,
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
