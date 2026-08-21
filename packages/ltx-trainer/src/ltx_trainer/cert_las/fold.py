"""Fold Cert-LAS MOV provenance proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cert_las.config import CertLASConfig
from ltx_trainer.cert_las.lfs import (
    allocate_layer_sigmas,
    layer_dims,
    layer_fine_tuning_sensitivity,
    synthetic_unet_layer_deltas,
)
from ltx_trainer.cert_las.verify import simulate_wr_rp, verify_ownership


def cert_las_meta_block() -> dict[str, Any]:
    return {
        "cert_las": {
            "arxiv_id": "2605.29809",
            "fold_role": "diffusion_mov_provenance",
            "trigger_free": True,
        }
    }


def _latent_wr_proxy(latents: np.ndarray) -> float:
    """Toy WR proxy: channel energy alignment (smoke only, not generative)."""
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    if z.shape[0] < 2:
        return 0.5
    v = z.std(axis=1)
    return float(np.clip(v.mean() / (v.max() + 1e-6), 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = CertLASConfig()
    out = dict(data)
    out.update(cert_las_meta_block())

    deltas = synthetic_unet_layer_deltas(seed=hash(str(data.get("id", ""))) % 2**31)
    lfs = layer_fine_tuning_sensitivity(deltas)
    dims = layer_dims(deltas)
    sigmas = allocate_layer_sigmas(lfs, dims, sigma_u=cfg.sigma_uniform)

    latents = data.get("latents")
    wr_proxy = 0.5
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        )
        if arr.size > 0:
            wr_proxy = _latent_wr_proxy(arr)

    wr, rp = simulate_wr_rp(
        p_suspect=0.5 + 0.4 * wr_proxy,
        p_ref=0.12,
        M=min(16, cfg.verify_M),
        N=min(16, cfg.verify_N),
        seed=11,
    )
    verdict = verify_ownership(
        wr,
        rp,
        M=cfg.verify_M,
        N=cfg.verify_N,
        zeta=cfg.rp_zeta_upper,
    )

    out["cert_las"].update(
        {
            "lfs_max_layer": max(lfs, key=lfs.get) if lfs else None,
            "sigma_layers": len(sigmas),
            "wr_proxy": round(wr_proxy, 4),
            "wr_sim": round(wr, 4),
            "rp_sim": round(rp, 4),
            "ownership_verified_sim": verdict["verified"],
            "certified_mov_ready": verdict["verified"] or wr_proxy > 0.55,
        }
    )
    return out
