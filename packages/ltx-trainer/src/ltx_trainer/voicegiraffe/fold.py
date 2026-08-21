"""Fold hour-scale audio QA sidecars into LTX audio save data."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig


def voicegiraffe_meta_block() -> dict[str, Any]:
    cfg = VoiceGiraffeConfig()
    return {
        "voicegiraffe": {
            "arxiv_id": "2605.27976",
            "fold_role": "long_context_aqa_sidecar",
            "avg_duration_min": cfg.avg_duration_min,
            "languages": list(cfg.languages),
            "n_sub_tasks": cfg.n_sub_tasks,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(voicegiraffe_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = VoiceGiraffeConfig()
    sr = float(data.get("sample_rate") or 48000.0)
    duration_min = arr.size / max(sr, 1.0) / 60.0
    long_context = duration_min >= 30.0

    out["voicegiraffe"].update(
        {
            "duration_min": round(duration_min, 2),
            "long_context_proxy": long_context,
            "hour_scale_proxy": duration_min >= 55.0,
        }
    )
    return out
