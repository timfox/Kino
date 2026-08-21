"""Fold WeatherProof degradation robustness readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    deg = float(meta.get("degradation_level") or meta.get("weather_severity") or 0.0)
    has_mask = data.get("weather_mask") is not None or meta.get("has_weather_mask")
    # Higher readiness when moderate degradation is labeled (trainable robustness)
    deg_score = float(np.clip(1.0 - abs(deg - 0.5) / 0.5, 0.0, 1.0)) if deg > 0 else (0.5 if has_mask else 0.25)
    mask_bonus = 0.25 if bool(has_mask) else 0.0
    readiness = float(np.clip(0.65 * deg_score + mask_bonus, 0.0, 1.0))
    out["weatherproof"] = {
        "robustness_readiness": round(readiness, 4),
        "degradation_level": deg,
        "has_weather_mask": bool(has_mask),
    }
    return out
