"""Fold Human-View survey readiness metadata for long-video MLLM clips."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    duration = float(meta.get("duration_sec") or meta.get("duration") or 0.0)
    is_long = duration >= 120.0 or bool(meta.get("long_video"))
    has_audio = bool(meta.get("has_audio") or meta.get("audio"))
    streaming = bool(meta.get("streaming") or meta.get("online"))
    reasoning = bool(meta.get("chain_of_thought") or meta.get("grounded_reasoning"))
    subfield = str(meta.get("video_subfield") or meta.get("domain") or "").lower()
    subfield_ok = any(
        tok in subfield
        for tok in ("ego", "sport", "instruction", "medical", "movie", "narrative")
    )
    readiness = float(
        np.clip(
            0.30 * float(is_long)
            + 0.20 * float(has_audio)
            + 0.20 * float(streaming)
            + 0.15 * float(reasoning)
            + 0.15 * float(subfield_ok),
            0.0,
            1.0,
        )
    )
    out["humanview_vu"] = {
        "arxiv_id": "2606.07433",
        "survey_readiness": round(readiness, 4),
        "is_long_video": is_long,
        "has_audio": has_audio,
        "streaming": streaming,
        "grounded_reasoning": reasoning,
        "subfield": subfield or None,
    }
    return out
