"""Fold streaming-diffusion music sidecars into LTX audio save data."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.demon.config import DemonConfig


def demon_meta_block() -> dict[str, Any]:
    cfg = DemonConfig()
    return {
        "demon": {
            "arxiv_id": "2605.28657",
            "fold_role": "streaming_music_sidecar",
            "base_model": cfg.base_model,
            "latent_hz": cfg.latent_hz,
            "production_depth": cfg.production_depth,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(demon_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = DemonConfig()
    sr = float(data.get("sample_rate") or 48000.0)
    duration_s = arr.size / max(sr, 1.0)
    frames = int(duration_s * cfg.latent_hz)

    rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
    out["demon"].update(
        {
            "duration_s": round(duration_s, 2),
            "latent_frames_proxy": frames,
            "rms": round(rms, 4),
            "streaming_tick_ms_anchor": cfg.tick_ms_depth4,
        }
    )
    return out
