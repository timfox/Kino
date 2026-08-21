"""Framework card, demos, and benchmark bundles for AirCast-SR."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.aircast_sr.config import AirCastSRConfig
from ltx_trainer.aircast_sr.interpolate import trilinear_time_spatial
from ltx_trainer.aircast_sr.lcm import fresh_denoiser, lcm_sample, training_mse_loss
from ltx_trainer.aircast_sr.layout import LIMITATIONS
from ltx_trainer.aircast_sr.metrics import skill_row
from ltx_trainer.aircast_sr.normalize import normalize_target_stack
from ltx_trainer.aircast_sr.patches import extract_patches, merge_patches
from ltx_trainer.aircast_sr.psd import radial_power_spectral_density
from ltx_trainer.aircast_sr.tables import (
    table_i_precipitation,
    table_ii_temperature,
    table_iii_surface_pressure,
    table_iv_humidity,
    table_v_longwave,
    table_vi_u_wind,
    table_vii_v_wind,
    table_viii_zero_shot_t2m,
)
from ltx_trainer.aircast_sr.variables import channel_spec


def framework_card(cfg: AirCastSRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AirCastSRConfig()
    return {
        "name": "AirCast-SR",
        "paper": cfg.paper_arxiv,
        "task": "Downscale GraphCast 0.25° (~28 km) → 1 km hourly, 67 h, 7 coupled surface variables.",
        "architecture": {
            "backbone": "UNet3DConditionModel",
            "diffusion": "Latent Consistency Model (LCM)",
            "in_channels": cfg.denoiser_in_channels,
            "out_channels": cfg.denoiser_out_channels,
            "block_channels": list(cfg.block_channels),
        },
        "conditioning": channel_spec(),
        "training": {
            "region": cfg.train_region,
            "year": cfg.train_year,
            "patch_size": cfg.train_patch_size,
            "optimizer": "AdamW",
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        "inference": {
            "patch_size": cfg.infer_patch_size,
            "stride": cfg.infer_stride,
            "lcm_steps": cfg.lcm_inference_steps,
            "gpu": "single commodity GPU / A100 minutes for CONUS",
        },
        "headline_properties": [
            "Near-zero systematic bias across variables and leads to 48 h",
            "Spectral fidelity 10–100 km (perception–distortion / structural realism)",
            "Zero-shot India / Germany (StationBench)",
        ],
        "limitations": list(LIMITATIONS),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    cfg = AirCastSRConfig()
    rng = np.random.default_rng(seed)
    t_cond, h, w = 5, 16, 16
    t_out = 8
    gc = rng.standard_normal((cfg.n_conditioning_channels, t_cond, h, w))
    gc_hourly = trilinear_time_spatial(gc, t_out=t_out, scale_h=1.0, scale_w=1.0)
    target_phys = rng.standard_normal((7, t_out, h, w)) * 0.5 + 0.5
    target = normalize_target_stack(target_phys)
    weights = fresh_denoiser(seed=seed, smoke=True)
    loss = training_mse_loss(target, gc_hourly, weights, timestep=400, rng=rng, cfg=cfg)
    pred = lcm_sample(gc_hourly, weights, shape=(t_out, h, w), rng=rng, cfg=cfg)
    t2m_idx = 1
    skill = skill_row(pred[t2m_idx, 0], target[t2m_idx, 0])
    wl, power = radial_power_spectral_density(pred[t2m_idx, 0], dx_km=1.0)
    patches = extract_patches(pred, patch_size=8, stride=4)
    merged = merge_patches(patches, out_shape=(h, w), patch_size=8)
    return {
        "train_mse": round(loss, 6),
        "t2m_skill": skill,
        "psd_bins": int(len(wl)),
        "patch_count": len(patches),
        "merge_shape": list(merged.shape),
        "lcm_steps": cfg.lcm_inference_steps,
        "gc_interp_shape": list(gc_hourly.shape),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_precipitation": table_i_precipitation(),
        "table_ii_temperature": table_ii_temperature(),
        "table_iii_surface_pressure": table_iii_surface_pressure(),
        "table_iv_humidity": table_iv_humidity(),
        "table_v_longwave": table_v_longwave(),
        "table_vi_u_wind": table_vi_u_wind(),
        "table_vii_v_wind": table_vii_v_wind(),
        "table_viii_zero_shot": table_viii_zero_shot_t2m(),
    }


def headline_results() -> dict[str, Any]:
    return {
        "resolution_km": 1.0,
        "forecast_hours": 67,
        "variables": 7,
        "winter_t2m_r_6h": 0.9716,
        "winter_t2m_bias_k": -0.007,
        "india_t2m_r_48h": 0.89,
        "near_zero_bias_all_vars": True,
    }
