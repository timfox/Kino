"""Fold PhysHDR-GS multi-exposure / splat readiness into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.physthdr_gs.config import EXPOSURE_TIMES, LDR_OE_INDICES


def physthdr_meta_block() -> dict[str, Any]:
    return {
        "physthdr_gs": {
            "arxiv_id": "2603.28020",
            "method": "3DGS + illumination-guided scaling + dual-branch tonemap",
            "exposure_times": list(EXPOSURE_TIMES),
            "ldr_oe_indices": list(LDR_OE_INDICES),
            "fold_role": "multi_exposure_gs_sidecar",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(physthdr_meta_block())

    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    ev_stack = data.get("hdr_ldr_ev_stack")
    ev_list = data.get("hdr_ev_list")

    num_exposures = len(EXPOSURE_TIMES)
    if ev_list is not None:
        num_exposures = len(ev_list) if hasattr(ev_list, "__len__") else num_exposures
    elif ev_stack is not None:
        arr = np.asarray(ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack)
        num_exposures = int(arr.shape[0]) if arr.ndim >= 1 else num_exposures

    under, over = 0.0, 0.0
    if ev_stack is not None:
        arr = np.asarray(ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack)
        flat = arr.reshape(arr.shape[0], -1).mean(axis=1)
        under = float(np.mean(flat <= 0.05))
        over = float(np.mean(flat >= 0.95))
    else:
        latents = data.get("latents")
        if latents is not None:
            arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
            flat = arr.reshape(-1)
            under = float(np.mean(flat <= np.quantile(flat, 0.05)))
            over = float(np.mean(flat >= np.quantile(flat, 0.95)))

    coverage = float(np.clip(num_exposures / len(EXPOSURE_TIMES), 0.0, 1.0))
    oe_need = float(np.clip(over * 0.6 + under * 0.4, 0.0, 1.0))
    readiness = float(np.clip(0.4 * coverage + 0.6 * oe_need, 0.0, 1.0))

    out["physthdr_gs"].update(
        {
            "gs_training_readiness": round(readiness, 4),
            "multi_exposure_coverage": round(coverage, 4),
            "highlight_clip_proxy": round(over, 4),
            "shadow_clip_proxy": round(under, 4),
            "num_exposures": int(num_exposures),
            "hdr_vae_encoding": hdr_meta.get("hdr_vae_encoding"),
        }
    )
    return out
