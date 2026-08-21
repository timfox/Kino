"""Fold DiffHDR log-gamma re-exposure readiness into video / HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.diffhdr.log_gamma import log_gamma_map
from ltx_trainer.hdr_ingest import diffhdr_meta_block


def _clip_fraction(arr: np.ndarray, *, low: float = 0.05, high: float = 0.95) -> tuple[float, float]:
    flat = arr.reshape(-1).astype(np.float64)
    return float(np.mean(flat <= low)), float(np.mean(flat >= high))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    encoding = str(hdr_meta.get("hdr_vae_encoding") or hdr_meta.get("vae_encoding") or "log_gamma")
    block = diffhdr_meta_block(vae_encoding=encoding)
    out.update(block)

    under, over = 0.0, 0.0
    dynamic = 0.0
    for key in ("hdr_linear", "sdr_preview", "latents"):
        val = data.get(key)
        if val is None:
            continue
        if hasattr(val, "detach"):
            import torch

            arr_t = val.detach().float().cpu()
            if key == "hdr_linear" and arr_t.numel() > 0:
                try:
                    lg = log_gamma_map(arr_t if arr_t.ndim >= 3 else arr_t.unsqueeze(0))
                    arr = lg.numpy()
                except Exception:
                    arr = arr_t.numpy()
            else:
                arr = arr_t.numpy()
        else:
            arr = np.asarray(val, dtype=np.float64)
        if arr.size == 0:
            continue
        sample = arr.reshape(-1, *arr.shape[-3:])[:1] if arr.ndim >= 4 else arr
        if sample.ndim == 3 and sample.shape[0] == 3:
            y = 0.2126 * sample[0] + 0.7152 * sample[1] + 0.0722 * sample[2]
        else:
            y = sample
        under, over = _clip_fraction(y)
        dynamic = float(np.clip(over + under, 0.0, 1.0))
        break

    encoding_ok = encoding.lower() in ("log_gamma", "log-gamma", "loggamma")
    readiness = float(np.clip((0.5 if encoding_ok else 0.2) + 0.5 * dynamic, 0.0, 1.0))
    out["diffhdr"].update(
        {
            "log_gamma_readiness": round(readiness, 4),
            "clip_under_fraction": round(under, 4),
            "clip_over_fraction": round(over, 4),
            "hdr_vae_encoding": encoding,
        }
    )
    return out
