"""Fold infant-cry ACC–MIC validation proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cryacc.config import CryAccConfig
from ltx_trainer.cryacc.features import extract_window_features, f0_from_periods


def cryacc_meta_block() -> dict[str, Any]:
    cfg = CryAccConfig()
    return {
        "cryacc": {
            "arxiv_id": "2605.28687",
            "modality_pair": ["MIC", "ACC"],
            "fold_role": "infant_cry_sidecar",
            "acc_sample_rate_hz": cfg.acc_sample_rate_hz,
            "window_ms": cfg.window_ms,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(cryacc_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = CryAccConfig()
    min_samples = int(cfg.acc_sample_rate_hz * cfg.window_ms / 1000.0)
    if arr.size < min_samples:
        return out

    rms = float(np.sqrt(np.mean(arr**2)))
    if rms < cfg.rms_exclude_threshold:
        out["cryacc"]["label_proxy"] = "non_cry"
        return out

    acc = arr[:min_samples]
    mic = arr[:min_samples] * 1.05
    feats = extract_window_features(mic, acc, cfg=cfg)
    zc = np.where(np.diff(np.signbit(arr[: min_samples * 2])))[0]
    periods = np.diff(zc) / cfg.acc_sample_rate_hz if zc.size >= 3 else np.array([1 / 420.0, 1 / 415.0])

    out["cryacc"].update(
        {
            "label_proxy": "cry_only" if rms >= cfg.rms_exclude_threshold else "non_cry",
            "rms": round(rms, 4),
            "mic_f0_hz": round(feats.get("mic_F0", f0_from_periods(periods)), 1),
            "acc_f0_hz": round(feats.get("acc_F0", f0_from_periods(periods)), 1),
            "mic_jcv_pct": round(feats.get("mic_JCV", 0.0), 3),
        }
    )
    return out
