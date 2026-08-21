"""Fold CAFNet half-truth detection proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig
from ltx_trainer.cafnet.features import extract_feature_triplet
from ltx_trainer.cafnet.model import cafnet_forward


def cafnet_meta_block() -> dict[str, Any]:
    cfg = CafNetConfig()
    return {
        "cafnet": {
            "arxiv_id": "2605.29531",
            "task": "ternary_deepfake + boundary_localisation",
            "features": ["mfcc", "lfcc", "chroma_stft"],
            "params": cfg.cafnet_params,
            "fold_role": "audio_integrity_sidecar",
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach CAFNet integrity proxy when waveform or features are present."""
    out = dict(data)
    out.update(cafnet_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    cfg = CafNetConfig()
    if arr.size < cfg.sample_rate_hz:
        arr = np.pad(arr, (0, cfg.sample_rate_hz - arr.size))
    feats = extract_feature_triplet(arr[: int(cfg.sample_rate_hz * cfg.clip_seconds)], cfg)
    pred = cafnet_forward(feats, cfg=cfg, seed=cfg.random_seed)
    probs = pred["class_probs"]
    out["cafnet"].update(
        {
            "pred_class": pred["pred_class"],
            "p_real": round(probs[0], 4),
            "p_fake": round(probs[1], 4),
            "p_half_truth": round(probs[2], 4),
            "boundary_start_s": round(pred["boundary_seconds"]["start"], 3),
            "boundary_end_s": round(pred["boundary_seconds"]["end"], 3),
        }
    )
    return out
