"""Mirage ablation variants (Tables 3–4)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.mirage.benchmarks import TABLE3_ABLATION, TABLE4_DEPTH, TABLE5_DOWNSAMPLE
from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.memory import LatentSpatialMemory, RGBPointCloudMemory
from ltx_trainer.mirage.rollout import run_toy_rollout


def ablation_variant(name: str) -> dict[str, float]:
    row = TABLE3_ABLATION.get(name)
    if row is None:
        raise KeyError(f"unknown ablation variant: {name}")
    return dict(row)


def depth_source_variant(name: str) -> dict[str, float]:
    mapping = {
        "depth_anything_3": "DepthAnything 3",
        "map_anything": "MapAnything",
        "unidepth": "UniDepth",
    }
    key = mapping.get(name, name)
    row = TABLE4_DEPTH.get(key)
    if row is None:
        raise KeyError(f"unknown depth source: {name}")
    return dict(row)


def run_ablation_smoke(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    full = ablation_variant("Mirage (full)")
    rgb = ablation_variant("Explicit RGB Point Cloud")
    rollout = run_toy_rollout(cfg, n_chunks=2, seed=1)
    mem = LatentSpatialMemory(channels=cfg.latent_channels)
    rgb_mem = RGBPointCloudMemory()
    return {
        "full_avg": full["avg"],
        "rgb_avg": rgb["avg"],
        "delta_avg_rgb": full["avg"] - rgb["avg"],
        "best_downsample": min(TABLE5_DOWNSAMPLE, key=lambda k: TABLE5_DOWNSAMPLE[k]),
        "rollout_memory_size": rollout["memory_size"],
        "latent_cache_bytes": mem.footprint_bytes(),
        "rgb_cache_bytes": rgb_mem.footprint_bytes(),
        "ok": full["avg"] > rgb["avg"],
    }
