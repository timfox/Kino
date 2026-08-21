"""Fold HoliTok holistic-token stats into LTX ``audio_latents`` shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.vae import compression_ratio, kl_gaussian


def holitok_meta_block(*, sample_rate: int = 48000) -> dict[str, Any]:
    cfg = HoliTokConfig()
    cr = compression_ratio(cfg.sample_rate_hz, cfg.latent_frame_rate_hz, cfg.latent_dim, bfloat_bits=cfg.bfloat_bits)
    return {
        "holitok": {
            "arxiv_id": "2605.29948",
            "latent_hz": cfg.latent_frame_rate_hz,
            "latent_dim": cfg.latent_dim,
            "compression_ratio": round(cr, 2),
            "variant": "Base",
            "fold_role": "audio_latent_sidecar",
        }
    }


def _latent_matrix(latents: Any) -> np.ndarray:
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    if arr.ndim == 1:
        cfg = HoliTokConfig()
        usable = (arr.size // cfg.latent_dim) * cfg.latent_dim
        if usable >= cfg.latent_dim:
            return arr[:usable].reshape(-1, cfg.latent_dim)
        return arr.reshape(1, -1)
    return arr.astype(np.float64)


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach HoliTok tokenizer proxies (KL, frame rate, holistic readiness)."""
    latents = data.get("latents")
    if latents is None:
        return {**data, **holitok_meta_block(sample_rate=int(data.get("sample_rate", 48000)))}

    z = _latent_matrix(latents)
    sr = int(data.get("sample_rate", 48000))
    cfg = HoliTokConfig()
    mean = z
    log_var = -2.0 * np.ones_like(z)
    kl = kl_gaussian(mean, log_var)
    frame_energy = float(np.mean(np.linalg.norm(z, axis=-1)))
    duration_s = z.shape[0] / cfg.latent_frame_rate_hz

    out = dict(data)
    out.update(holitok_meta_block(sample_rate=sr))
    out["holitok"].update(
        {
            "kl_proxy": round(kl, 4),
            "frame_energy_mean": round(frame_energy, 4),
            "num_latent_frames": int(z.shape[0]),
            "duration_s": round(duration_s, 3),
            "holistic_readiness_proxy": round(float(np.clip(1.0 - kl / 10.0, 0.0, 1.0)), 4),
        }
    )
    return out
