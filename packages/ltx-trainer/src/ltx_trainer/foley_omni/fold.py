"""Audio latent sidecar: Foley-Omni V2ST sync / structured soundtrack proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.foley_omni.config import FoleyOmniConfig


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    cfg = FoleyOmniConfig()
    enc = data.get("audio_encode") if isinstance(data.get("audio_encode"), dict) else {}
    caption = str(data.get("caption") or data.get("prompt") or "")
    has_words = "[WORDS]" in caption or "[words]" in caption.lower()
    has_audio_tag = "[AUDIO]" in caption or "[audio]" in caption.lower()
    sync_off = int(enc.get("sync_offset_frames") or 0)
    sync_penalty = float(np.clip(abs(sync_off) / 8.0, 0.0, 1.0))
    out["foley_omni_audio"] = {
        "arxiv": cfg.paper_arxiv,
        "fold_role": "v2st_sync_proxy",
        "structured_caption": has_words or has_audio_tag,
        "sync_offset_frames": sync_off,
        "sync_score": round(1.0 - sync_penalty, 4),
        "desync_risk": sync_penalty > 0.35,
        "video_fps": enc.get("video_fps"),
        "duration_sec": enc.get("duration_sec"),
        "v2st_bench_ready": bool(caption.strip()),
    }
    return out
