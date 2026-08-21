"""Single-video VideoMAE steering check (baseline vs ±α)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ltx_trainer.physics_steering.bundle import load_steering_bundle
from ltx_trainer.physics_steering.collect import forward_videomae_pooled, load_video_tensor
from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.inference_hooks import PhysicsSteeringHookManager
from ltx_trainer.physics_steering.steering import probe_impossible_score


def check_video_steering(
    video_path: str | Path,
    bundle_path: str | Path,
    *,
    cfg: PhysicsSteeringConfig | None = None,
    device: str | None = None,
    alphas: tuple[float, ...] = (-5.0, 0.0, 5.0),
) -> dict[str, Any]:
    """Encode one clip; compare pooled PEZ features with and without CAV hooks."""
    cfg = cfg or PhysicsSteeringConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_steering_bundle(bundle_path)
    probe_layer = int(bundle["probe_layer"])
    w, b = bundle["probe_weight"].to(device), bundle["probe_bias"]
    cav_by_layer = {k: v.to(device) for k, v in bundle["cav_by_layer"].items()}

    from ltx_trainer.physics_steering.videomae_hooks import load_videomae_model

    model, processor = load_videomae_model(cfg)
    model = model.to(device).eval()
    pixels = load_video_tensor(
        Path(video_path),
        num_frames=cfg.num_frames,
        spatial_size=cfg.spatial_size,
    ).to(device)

    baseline = forward_videomae_pooled(pixels, model, processor, num_layers=cfg.num_layers)
    f0 = baseline[probe_layer].to(device)
    p_base = float(probe_impossible_score(f0.unsqueeze(0), w, b).item())

    rows: list[dict[str, float]] = [{"alpha": 0.0, "p_impossible": p_base}]
    for alpha in alphas:
        if alpha == 0.0:
            continue
        mgr = PhysicsSteeringHookManager(model, cav_by_layer, alpha=alpha, num_layers=cfg.num_layers)
        mgr.register_steering()
        steered = forward_videomae_pooled(pixels, model, processor, num_layers=cfg.num_layers)
        mgr.remove_steering()
        f1 = steered[probe_layer]
        p1 = float(probe_impossible_score(f1.unsqueeze(0), w, b).item())
        rows.append({"alpha": float(alpha), "p_impossible": p1})

    return {
        "video": str(video_path),
        "probe_layer": probe_layer,
        "baseline_p_impossible": p_base,
        "alpha_rows": rows,
        "bundle": str(bundle_path),
    }
