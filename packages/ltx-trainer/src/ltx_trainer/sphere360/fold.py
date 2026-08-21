"""AV-fold sidecar for Sphere360 HDR timelapse readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    clip_id = meta.get("sphere360_clip_id") or meta.get("clip_id") or hdr_meta.get("sphere360_clip_id")
    has_foa = bool(meta.get("has_foa_audio") or meta.get("foa_audio"))
    is_erp = bool(meta.get("erp") or meta.get("is_360") or hdr_meta.get("projection") == "equirect")
    duration = float(meta.get("duration_sec") or hdr_meta.get("duration_sec") or 0.0)
    dur_score = float(np.clip(duration / 10.0, 0.0, 1.0)) if duration > 0 else (0.75 if clip_id else 0.4)
    id_score = 1.0 if clip_id else 0.0
    foa_bonus = 0.15 if has_foa else 0.0
    erp_bonus = 0.2 if is_erp else 0.0
    readiness = float(np.clip(0.45 * id_score + 0.25 * dur_score + foa_bonus + erp_bonus, 0.0, 1.0))
    out["sphere360"] = {
        "hub": "omniaudio/Sphere360",
        "timelapse_readiness": round(readiness, 4),
        "clip_id": str(clip_id) if clip_id else None,
        "has_foa_audio": has_foa,
        "is_equirect": is_erp,
        "duration_sec": duration,
    }
    return out
