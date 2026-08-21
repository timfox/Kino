"""Benchmark catalog outline (counts and taxonomy; full JSON is upstream)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.config import LongAVCompassConfig


def benchmark_catalog_outline(cfg: LongAVCompassConfig | None = None) -> dict[str, Any]:
    """Structural catalog for agents planning eval runs (not per-case IDs)."""
    cfg = cfg or LongAVCompassConfig()
    return {
        "total_samples": cfg.n_samples,
        "tasks": {
            "T2AV": {
                "samples": cfg.n_t2av,
                "events": cfg.n_events_t2av,
                "shots": cfg.n_shots_t2av,
                "conditioning": "script only",
            },
            "I2AV": {
                "samples": cfg.n_i2av,
                "events": cfg.n_events_i2av,
                "shots": cfg.n_shots_i2av,
                "conditioning": "reference image + script",
            },
            "V2AV": {
                "samples": cfg.n_v2av,
                "events": cfg.n_events_v2av,
                "shots": cfg.n_shots_v2av,
                "conditioning": "reference video (10–15s) + continuation script",
            },
        },
        "scenarios": ["Vlog", "Content-Creator", "Performance Ads", "Brand Ads"],
        "complexity_levels": ["L1", "L2", "L3", "L4"],
        "avg_events_per_sample": {
            "T2AV": round(cfg.n_events_t2av / cfg.n_t2av, 2),
            "I2AV": round(cfg.n_events_i2av / cfg.n_i2av, 2),
            "V2AV": round(cfg.n_events_v2av / cfg.n_v2av, 2),
        },
        "bundled_examples": [
            "t2av_perf_ads_l4_demo",
            "i2av_perf_ads_l4_watch",
            "v2av_content_creator_l4_missed_connection",
        ],
        "upstream_layout": {
            "root": "<release_dir>/",
            "per_case": [
                "cases/<case_id>/annotation.json",
                "cases/<case_id>/reference_image.jpg",  # I2AV
                "cases/<case_id>/reference_video.mp4",  # V2AV
            ],
            "per_model_output": [
                "<model>/<case_id>/full_video.mp4",
                "<model>/<case_id>/canonical_events.json",
                "<model>/<case_id>/events/event_XX.mp4",
                "<model>/<case_id>/boundaries/boundary_XX_YY.mp4",
            ],
        },
    }
