"""Fold BTS-CAFE respiratory-device proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig
from ltx_trainer.btscafe.gin import gin_augment
from ltx_trainer.btscafe.text_aug import counterfactual_text_augment


def btscafe_meta_block() -> dict[str, Any]:
    cfg = BTSCafeConfig()
    return {
        "btscafe": {
            "arxiv_id": "2605.29862",
            "task": "federated_rsc_device_generalization",
            "fold_role": "respiratory_audio_sidecar",
            "sample_rate_hz": cfg.sample_rate_hz,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(btscafe_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    if arr.size < 512:
        return out

    cfg = BTSCafeConfig()
    hop = max(256, arr.size // 64)
    frames = []
    for start in range(0, arr.size - hop, hop):
        chunk = arr[start : start + hop]
        frames.append(np.abs(np.fft.rfft(chunk))[:32])
    spec = np.stack(frames[:64], axis=1) if frames else np.zeros((32, 1))
    if spec.shape[1] < spec.shape[0]:
        spec = np.pad(spec, ((0, 0), (0, spec.shape[0] - spec.shape[1])), mode="edge")

    gin = gin_augment(spec, cfg=cfg, seed=int(np.sum(np.abs(arr[:64])) % 10_000))
    meta = data.get("text_prompt") or data.get("caption") or ""
    if meta:
        text = counterfactual_text_augment(str(meta), cfg=cfg, seed=42)
        out["btscafe"]["counterfactual_text"] = text["augmented"][:200]

    out["btscafe"].update(
        {
            "gin_alpha_mean": gin["alpha_mean"],
            "device_style_proxy_applied": True,
        }
    )
    return out
