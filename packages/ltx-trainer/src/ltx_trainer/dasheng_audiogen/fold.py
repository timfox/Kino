"""Fold mixed-audio scene sidecars into LTX audio save data."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig


def dasheng_audiogen_meta_block() -> dict[str, Any]:
    cfg = DashengAudioGenConfig()
    return {
        "dasheng_audiogen": {
            "arxiv_id": "2605.27838",
            "fold_role": "mixed_audio_scene_sidecar",
            "latent_dim": cfg.latent_dim,
            "latent_hz": cfg.latent_hz,
            "clip_duration_s": cfg.clip_duration_s,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(dasheng_audiogen_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = DashengAudioGenConfig()
    sr = float(data.get("sample_rate") or 48000.0)
    duration_s = arr.size / max(sr, 1.0)
    frames = int(min(duration_s, cfg.clip_duration_s) * cfg.latent_hz)

    out["dasheng_audiogen"].update(
        {
            "duration_s": round(duration_s, 2),
            "latent_frames_proxy": frames,
            "ten_second_clip": duration_s <= cfg.clip_duration_s + 0.5,
        }
    )
    return out
