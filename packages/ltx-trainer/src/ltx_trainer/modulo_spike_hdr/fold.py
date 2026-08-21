"""Fold modulo-spike HDR unwrap readiness into video / sensor latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def modulo_spike_meta_block(*, period: float = 256.0) -> dict[str, Any]:
    return {
        "modulo_spike_hdr": {
            "arxiv_id": "2604.14632",
            "method": "PMF-Adapter + LMA-Decoder + CCP-Refiner",
            "modulo_period": period,
            "effective_fps_paper": 1000.0,
            "fold_role": "spike_modulo_unwrap_sidecar",
        }
    }


def _saturation_proxy(arr: np.ndarray) -> float:
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0
    p95 = float(np.quantile(flat, 0.95))
    p05 = float(np.quantile(flat, 0.05))
    return float(np.clip((p95 - p05) * 2.0, 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    period = float(hdr_meta.get("modulo_period", 256.0))
    out.update(modulo_spike_meta_block(period=period))

    wrap_energy = 0.0
    for key in ("hdr_linear", "latents", "sdr_preview"):
        val = data.get(key)
        if val is None:
            continue
        arr = np.asarray(val.detach().cpu().float().numpy() if hasattr(val, "detach") else val)
        if arr.size == 0:
            continue
        wrap_energy = _saturation_proxy(arr)
        break

    # High dynamic range / folding-prone regions benefit from modulo unwrap.
    readiness = float(np.clip(0.25 + 0.75 * wrap_energy, 0.0, 1.0))
    out["modulo_spike_hdr"].update(
        {
            "unwrap_readiness_proxy": round(readiness, 4),
            "modulo_wrap_energy": round(wrap_energy, 4),
            "spike_encode_ready": wrap_energy > 0.15,
            "hdr_vae_encoding": hdr_meta.get("hdr_vae_encoding"),
        }
    )
    return out
