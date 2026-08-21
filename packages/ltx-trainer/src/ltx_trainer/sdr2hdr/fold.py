"""Fold SDR2HDR MEVM/VMM merge readiness into video / HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import sdr2hdr_meta_block
from ltx_trainer.sdr2hdr.mevm_proxy import DEFAULT_MEVM_EVS


def _temporal_stability(z: np.ndarray) -> float:
    if z.ndim < 2 or z.shape[0] < 2:
        return 0.5
    flat = z.reshape(z.shape[0], -1).mean(axis=1)
    return float(np.mean(np.abs(np.diff(flat))))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    evs = hdr_meta.get("mevm_evs") or list(DEFAULT_MEVM_EVS)
    block = sdr2hdr_meta_block(mevm_evs=tuple(float(x) for x in evs[:3]))
    out.update(block)

    sdr = data.get("sdr_preview") or data.get("latents")
    stability = 0.5
    if sdr is not None:
        arr = np.asarray(sdr.detach().cpu().float().numpy() if hasattr(sdr, "detach") else sdr)
        if arr.ndim >= 3:
            stability = _temporal_stability(arr)
    has_hdr = data.get("hdr_linear") is not None
    if not has_hdr:
        meta_flag = hdr_meta.get("has_hdr")
        has_hdr = bool(meta_flag) if not hasattr(meta_flag, "numel") else bool(meta_flag.any().item())
    pair_score = 1.0 if has_hdr else 0.55
    merge_readiness = float(np.clip(pair_score * (1.0 - stability * 2.0), 0.0, 1.0))

    out["sdr2hdr"].update(
        {
            "merge_readiness_proxy": round(merge_readiness, 4),
            "temporal_stability_proxy": round(stability, 5),
            "mevm_evs": list(evs)[:3],
            "has_sdr_hdr_pair": bool(has_hdr),
        }
    )
    return out
