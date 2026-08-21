"""Demos and ablation checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.spherediff.benchmarks import TABLE3_ABLATION, ours_beats_dynamic_scaler
from ltx_trainer.spherediff.config import SphereDiffConfig
from ltx_trainer.spherediff.fibonacci import fibonacci_sphere
from ltx_trainer.spherediff.metrics import end_continuity_score, seam_quality_score
from ltx_trainer.spherediff.spherediff_net import SphereDiffStub


def evaluation_demo_run(cfg: SphereDiffConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SphereDiffConfig(num_latents=128, num_views=6, denoise_steps=2)
    model = SphereDiffStub(cfg)
    with torch.no_grad():
        out = model(["upper sky", "middle scene", "lower ground", "middle", "upper"])
    dirs = fibonacci_sphere(64)
    return {
        "erp_shape": list(out["erp_rgb"].shape),
        "num_latents_cfg": cfg.num_latents,
        "seam_score": float(seam_quality_score(out["erp_rgb"]).item()),
        "beats_dynamic_scaler_distortion": ours_beats_dynamic_scaler("distortion"),
        "table3_full_distortion": TABLE3_ABLATION["Dynamic + Weighted Avg."]["distortion"],
    }


def train_step(cfg: SphereDiffConfig | None = None) -> dict[str, float]:
    cfg = cfg or SphereDiffConfig(num_latents=64, num_views=4, denoise_steps=1)
    model = SphereDiffStub(cfg)
    out = model()
    loss = out["erp_rgb"].mean()
    loss.backward()
    return {"loss": float(loss.detach())}


def ablation_checks() -> dict[str, bool]:
    full = TABLE3_ABLATION["Dynamic + Weighted Avg."]
    near = TABLE3_ABLATION["Nearest sampling"]
    return {
        "dynamic_weighted_best_distortion": full["distortion"] > near["distortion"],
        "weighted_helps_nearest": TABLE3_ABLATION["Nearest + Weighted Avg."]["end_continuity"]
        > near["end_continuity"],
        "dynamic_beats_nearest_alone": TABLE3_ABLATION["Dynamic sampling"]["end_continuity"]
        > near["end_continuity"],
    }
