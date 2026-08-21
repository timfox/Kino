"""Fold LumiVid LogC3 alignment readiness into HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import lumivid_meta_block


def _clip_fraction(arr: np.ndarray, *, low: float = 0.02, high: float = 0.98) -> tuple[float, float]:
    flat = arr.reshape(-1).astype(np.float64)
    under = float(np.mean(flat <= low))
    over = float(np.mean(flat >= high))
    return under, over


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    encoding = str(hdr_meta.get("hdr_vae_encoding") or hdr_meta.get("vae_encoding") or "logc3")

    block = lumivid_meta_block(vae_encoding=encoding)
    out.update(block)

    under, over = 0.0, 0.0
    for key in ("latents", "sdr_preview", "hdr_linear"):
        val = data.get(key)
        if val is None:
            continue
        arr = np.asarray(val.detach().cpu().float().numpy() if hasattr(val, "detach") else val)
        if arr.size == 0:
            continue
        sample = arr
        if arr.ndim >= 4:
            sample = arr.reshape(-1, *arr.shape[-3:])[:1]
        if sample.ndim == 3 and sample.shape[0] == 3:
            y = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]
        else:
            y = sample
        under, over = _clip_fraction(y)
        break

    encoding_ok = encoding.lower() in ("logc3", "log_c3", "log-c3")
    dynamic_range = float(np.clip(over + under, 0.0, 1.0))
    readiness = float(np.clip((0.55 if encoding_ok else 0.2) + 0.45 * dynamic_range, 0.0, 1.0))

    out["lumivid"].update(
        {
            "logc3_alignment_readiness": round(readiness, 4),
            "clip_under_fraction": round(under, 4),
            "clip_over_fraction": round(over, 4),
            "needs_synthesis_fraction": round(float(np.clip(over * 0.7 + under * 0.3, 0.0, 1.0)), 4),
            "hdr_vae_encoding": encoding,
        }
    )
    return out
