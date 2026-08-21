"""LongAV-Compass metadata on latent shards (minute-scale routing hint, no train weight)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.config import LongAVCompassConfig


def longav_compass_meta_block() -> dict[str, Any]:
    return {
        "longav_compass": {
            "arxiv_id": "2605.26244",
            "fold_role": "minute_scale_eval_routing_hint",
            "note": "Does not change diffusion loss; use for delivery / eval planning only.",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Tag shards that may suit minute-scale LongAV-style eval vs short AVBench slices."""
    cfg = LongAVCompassConfig()
    out = dict(data)
    out.update(longav_compass_meta_block())

    num_frames = data.get("num_frames")
    fps = float(data.get("fps", 24.0) or 24.0)
    if num_frames is None:
        out["longav_compass"]["estimated_duration_s"] = None
        out["longav_compass"]["minute_scale_candidate"] = False
        return out

    try:
        nf = int(num_frames)
    except (TypeError, ValueError):
        nf = 0
    duration_s = nf / max(fps, 1.0)
    out["longav_compass"].update(
        {
            "estimated_duration_s": round(duration_s, 2),
            "minute_scale_candidate": duration_s >= float(cfg.target_duration_s_min),
            "target_duration_s_range": [cfg.target_duration_s_min, cfg.target_duration_s_max],
            "benchmark_samples": cfg.n_samples,
        }
    )
    return out
