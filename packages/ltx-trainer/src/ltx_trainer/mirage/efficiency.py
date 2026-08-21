"""Efficiency scaling vs RGB-cache baselines (Fig. 5, arXiv:2606.09828)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.memory import RGBPointCloudMemory, latent_readout_cost


FIG5_READOUT_S_PER_FRAME: dict[str, list[float]] = {
    "Mirage": [0.79, 0.25, 0.25, 0.25, 0.25],
    "Gen3C": [2.80, 2.88, 2.88, 2.87, 2.90],
    "Spatia": [4.03, 5.03, 6.32, 6.00, 7.63],
    "VMem": [4.07, 6.19, 6.79, 7.24, 7.25],
}

FIG5_CACHE_VRAM_MIB: dict[str, list[float]] = {
    "Mirage": [0.46, 0.90, 1.36, 1.80, 2.25],
    "Gen3C": [22.3, 43.8, 43.8, 43.8, 43.8],
    "Spatia": [23.4, 47.0, 73.2, 98.4, 124.0],
    "VMem": [18.8, 33.8, 48.8, 63.8, 78.8],
}


def figure5_scaling() -> dict[str, Any]:
    return {"readout_s_per_frame": FIG5_READOUT_S_PER_FRAME, "cache_vram_mib": FIG5_CACHE_VRAM_MIB}


def asymptotic_readout_cost(
    n_points: int,
    *,
    rgb_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    vae_stride: int = 16,
) -> dict[str, float]:
    """Appendix B: Θ(N log N + HW) + Φ_E vs Θ(N log N + hw)."""
    H, W = rgb_hw
    h, w = latent_hw
    n = max(n_points, 2)
    sort_term = n * np.log2(n)
    rgb_total = sort_term + H * W + H * W  # raster + VAE encode
    latent_total = sort_term + h * w
    mem_rgb = n * (3 * 8)
    mem_lat = n * (3 * 8 + 48 * 8)  # coords + C=48 features (approx)
    return {
        "rgb_readout": float(rgb_total),
        "latent_readout": float(latent_total),
        "readout_ratio": float(rgb_total / max(latent_total, 1e-9)),
        "cache_memory_ratio": float((mem_rgb) / max(mem_lat / (vae_stride**2 * 48 / 3), 1e-9)),
        "latent_hw": float(h * w),
        "rgb_hw": float(H * W),
        "stride_sq": float(vae_stride**2),
    }


def model_cost_ratio(cfg: MirageConfig | None = None) -> dict[str, float]:
    """Analytic cost ratio latent vs RGB readout at paper grid sizes."""
    cfg = cfg or MirageConfig()
    rng = np.random.default_rng(0)
    rgb = RGBPointCloudMemory()
    for _ in range(5000):
        rgb.points.append(rng.standard_normal(3))
        rgb.colors.append(rng.random(3))
    rgb_cost = rgb.estimated_readout_cost(rgb_hw=cfg.rgb_grid, latent_hw=cfg.latent_grid)["total"]
    lat_cost = latent_readout_cost(len(rgb.points), cfg.latent_grid)["total"]
    mem_rgb = rgb.footprint_bytes()
    mem_lat = len(rgb.points) * (3 * 8 + cfg.latent_channels * 8)
    return {
        "readout_speedup": float(rgb_cost / max(lat_cost, 1e-9)),
        "memory_reduction": float(mem_rgb / max(mem_lat, 1)),
        "paper_readout_speedup": cfg.speedup_vs_rgb,
        "paper_memory_reduction": cfg.memory_reduction_vs_rgb,
    }


def efficiency_demo(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    fig5 = figure5_scaling()
    ratio = model_cost_ratio(cfg)
    mirage_steady = fig5["readout_s_per_frame"]["Mirage"][-1]
    spatia_steady = fig5["readout_s_per_frame"]["Spatia"][-1]
    return {
        "figure5": fig5,
        "steady_state_s_per_frame": {"Mirage": mirage_steady, "Spatia": spatia_steady},
        "speedup_vs_spatia": float(spatia_steady / mirage_steady),
        "cost_model": ratio,
        "asymptotic": asymptotic_readout_cost(5000, rgb_hw=cfg.rgb_grid, latent_hw=cfg.latent_grid, vae_stride=cfg.vae_stride),
        "mirage_cache_mib_final_chunk": fig5["cache_vram_mib"]["Mirage"][-1],
        "spatia_cache_mib_final_chunk": fig5["cache_vram_mib"]["Spatia"][-1],
    }
