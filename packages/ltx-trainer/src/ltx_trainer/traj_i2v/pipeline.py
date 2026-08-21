"""Trajectory-guided maritime video reconstruction pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.traj_i2v.baselines import optical_flow_extrapolate, rife_interpolate_stub
from ltx_trainer.traj_i2v.conditioning import SgI2VConditioning, build_sg_i2v_conditioning
from ltx_trainer.traj_i2v.config import TrajI2VConfig
from ltx_trainer.traj_i2v.gps_mapping import GpsFix, VesselAnchor, estimate_scale_px_per_m, mean_origin
from ltx_trainer.traj_i2v.metrics import (
    brisque_proxy,
    mean_flow_magnitude,
    mse_lpips_proxy,
    trajectory_error_px,
)
from ltx_trainer.traj_i2v.synthetic import (
    synthetic_anchors,
    synthetic_gps_log,
    synthetic_reference_frame,
)


def sg_i2v_generate_stub(
    conditioning: SgI2VConditioning,
    *,
    cfg: TrajI2VConfig,
) -> Tensor:
    """Stub generator: motion-blur reference along GPS trajectories (no external SG-I2V weights)."""
    base = conditioning.reference_frame_annotated
    if base is None:
        raise ValueError("conditioning missing annotated reference")
    frames: list[Tensor] = []
    for i in range(cfg.num_frames):
        t = (i + 1) / cfg.num_frames
        f = base.clone()
        for obj in conditioning.objects:
            if obj.name in ("green", "yellow"):
                pt = obj.trajectory[i]
                cx, cy = int(pt[0].item()), int(pt[1].item())
                h, w = f.shape[-2:]
                if 0 <= cx < w and 0 <= cy < h:
                    f[:, cy, cx] = f[:, cy, cx] * (1 - 0.1 * t) + torch.tensor([1.0, 1.0, 1.0]) * 0.1 * t
        frames.append(f.clamp(0, 1))
    return torch.stack(frames, dim=0)


def reconstruct_clip(
    reference: Tensor,
    log_by_vessel: dict[int, list[GpsFix]],
    anchors: list[VesselAnchor],
    *,
    cfg: TrajI2VConfig | None = None,
    method: str = "sg_i2v",
    ground_truth: Tensor | None = None,
) -> dict[str, Any]:
    """Run one reconstruction method; return frames + metric dict."""
    cfg = cfg or TrajI2VConfig()
    all_fixes = [f for fixes in log_by_vessel.values() for f in fixes]
    origin_lon, origin_lat = mean_origin(all_fixes)
    scale = estimate_scale_px_per_m(anchors[0], anchors[1], origin_lon, origin_lat)
    conditioning = build_sg_i2v_conditioning(
        reference,
        log_by_vessel,
        anchors,
        cfg=cfg,
        scale_px_per_m=scale,
    )

    if method == "sg_i2v":
        frames = sg_i2v_generate_stub(conditioning, cfg=cfg)
    elif method == "optical_flow":
        frames = optical_flow_extrapolate(reference, num_frames=cfg.num_frames)
    elif method == "rife":
        end = ground_truth[0] if ground_truth is not None and ground_truth.shape[0] > 0 else reference
        frames = rife_interpolate_stub(reference, end, num_frames=cfg.num_frames)
    else:
        raise ValueError(f"unknown method {method!r}")

    stats: dict[str, Any] = {
        "method": method,
        "num_frames": frames.shape[0],
        "temporal_smoothness": mean_flow_magnitude(frames),
        "brisque": brisque_proxy(frames),
    }
    if ground_truth is not None and ground_truth.shape[0] == frames.shape[0]:
        stats["lpips"] = mse_lpips_proxy(frames, ground_truth)
    # GPS conditioning adherence proxy: mean step size vs first projected point (px)
    errs: list[float] = []
    for obj in conditioning.objects:
        if obj.name in ("green", "yellow"):
            t0 = obj.trajectory[0:1]
            errs.append(trajectory_error_px(obj.trajectory, t0.expand_as(obj.trajectory)))
    if errs:
        stats["trajectory_error_px"] = sum(errs) / len(errs)
    return {"frames": frames, "conditioning": conditioning, "metrics": stats, "scale_px_per_m": scale}


def evaluation_demo_run(*, device: str = "cpu") -> dict[str, Any]:
    cfg = TrajI2VConfig(image_width=128, image_height=72, num_frames=8)
    ref = synthetic_reference_frame(cfg).to(device)
    log = synthetic_gps_log(cfg)
    anchors = synthetic_anchors(cfg)
    out_sg = reconstruct_clip(ref, log, anchors, cfg=cfg, method="sg_i2v")
    out_of = reconstruct_clip(ref, log, anchors, cfg=cfg, method="optical_flow")
    return {
        "config": {k: v for k, v in asdict(cfg).items()},
        "sg_i2v_frames": int(out_sg["frames"].shape[0]),
        "optical_flow_temporal": out_of["metrics"]["temporal_smoothness"],
        "sg_i2v_temporal": out_sg["metrics"]["temporal_smoothness"],
        "num_conditioning_objects": len(out_sg["conditioning"].objects),
    }


def save_conditioning_json(conditioning: SgI2VConditioning, path: Path | str) -> None:
    import json

    payload = []
    for obj in conditioning.objects:
        payload.append(
            {
                "name": obj.name,
                "box": obj.box.tolist(),
                "trajectory": obj.trajectory.tolist(),
            }
        )
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
