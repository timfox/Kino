"""Fold ChildVox developmental-audio proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.childvox.config import ChildVoxConfig
from ltx_trainer.childvox.encoder_proxy import layer_hiddens_from_waveform
from ltx_trainer.childvox.models import weighted_encoder_pool


def childvox_meta_block() -> dict[str, Any]:
    cfg = ChildVoxConfig()
    return {
        "childvox": {
            "arxiv_id": "2605.29257",
            "categories": ["physiological", "vocalization", "canonical_syllables", "speech"],
            "fold_role": "child_audio_sidecar",
            "sample_rate_hz": cfg.sample_rate_hz,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(childvox_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = ChildVoxConfig()
    if arr.size < cfg.sample_rate_hz // 5:
        return out

    rng = np.random.default_rng(int(np.sum(np.abs(arr[: min(512, arr.size)])) * 1e6) % 2**32)
    hiddens = layer_hiddens_from_waveform(arr, sample_rate=cfg.sample_rate_hz)
    pred = weighted_encoder_pool(hiddens, num_classes=5, seed=int(rng.integers(0, 2**31)))
    out["childvox"].update(
        {
            "vocal_dev_proxy_class": pred["pred"],
            "child_speech_ready": arr.size >= cfg.sample_rate_hz * cfg.min_duration_ms / 1000,
        }
    )
    return out
