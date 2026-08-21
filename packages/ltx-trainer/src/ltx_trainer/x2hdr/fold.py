"""Fold X2HDR PU21 alignment readiness into HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import X2HDR_DEFAULT_L_PEAK_CD_M2, x2hdr_meta_block


def _clip_fraction(arr: np.ndarray, *, low: float = 0.02, high: float = 0.98) -> tuple[float, float]:
    flat = arr.reshape(-1).astype(np.float64)
    return float(np.mean(flat <= low)), float(np.mean(flat >= high))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    l_peak = float(hdr_meta.get("L_peak_cd_m2") or hdr_meta.get("peak_nits") or X2HDR_DEFAULT_L_PEAK_CD_M2)
    encoding = str(hdr_meta.get("hdr_vae_encoding") or hdr_meta.get("vae_encoding") or "pu21")

    block = x2hdr_meta_block(l_peak=l_peak, vae_encoding=encoding)
    out.update(block)

    under, over = 0.0, 0.0
    has_sdr = False
    for key in ("sdr_preview", "latents", "hdr_linear"):
        val = data.get(key)
        if val is None:
            continue
        arr = np.asarray(val.detach().cpu().float().numpy() if hasattr(val, "detach") else val)
        if arr.size == 0:
            continue
        sample = arr.reshape(-1, *arr.shape[-3:])[:1] if arr.ndim >= 4 else arr
        if sample.ndim == 3 and sample.shape[0] == 3:
            y = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]
        else:
            y = sample
        under, over = _clip_fraction(y)
        has_sdr = key == "sdr_preview" or has_sdr
        break

    encoding_ok = encoding.lower() in ("pu21", "x2hdr")
    peak_ok = min(1.0, l_peak / X2HDR_DEFAULT_L_PEAK_CD_M2)
    dynamic = float(np.clip(over + under, 0.0, 1.0))
    pair_bonus = 0.15 if has_sdr or data.get("sdr_path") else 0.0
    readiness = float(np.clip((0.45 if encoding_ok else 0.15) + 0.35 * dynamic + 0.2 * peak_ok + pair_bonus, 0.0, 1.0))

    out["x2hdr"].update(
        {
            "pu21_alignment_readiness": round(readiness, 4),
            "clip_under_fraction": round(under, 4),
            "clip_over_fraction": round(over, 4),
            "hdr_vae_encoding": encoding,
            "L_peak_cd_m2": l_peak,
        }
    )
    return out
