"""Fold FMelCodec mel-coding stats into LTX ``audio_latents`` shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fmelcodec.coding import coding_stage_loss
from ltx_trainer.fmelcodec.config import FMelCodecConfig
from ltx_trainer.fmelcodec.oc_vq import bitrate_bps, quantize_nearest


def fmelcodec_meta_block(*, sample_rate: int = 16000) -> dict[str, Any]:
    cfg = FMelCodecConfig()
    return {
        "fmelcodec": {
            "arxiv_id": "2605.25669",
            "target_bitrate_bps": cfg.bitrate_16k_bps if sample_rate <= 24000 else cfg.bitrate_48k_bps,
            "codebook_size": cfg.codebook_size,
            "latent_dim": cfg.latent_dim,
            "fold_role": "audio_latent_sidecar",
        }
    }


def _latent_proxy_vector(latents: Any, *, dim: int, seed: int = 0) -> np.ndarray:
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1)
    if flat.size < dim:
        flat = np.pad(flat, (0, dim - flat.size))
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=dim, replace=flat.size < dim)
    return flat[idx].astype(np.float64)


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach OC-VQ proxy coding loss + bitrate estimate (numpy; no extra tensors)."""
    cfg = FMelCodecConfig()
    latents = data.get("latents")
    if latents is None:
        return {**data, **fmelcodec_meta_block()}

    z = _latent_proxy_vector(latents, dim=cfg.latent_dim)
    rng = np.random.default_rng(int(abs(z.sum() * 1e4)) % (2**31))
    codebook = rng.standard_normal((cfg.codebook_size, cfg.latent_dim))
    codebook /= np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-8
    _idx, z_q = quantize_nearest(z, codebook)
    natural = z.reshape(1, -1)
    coarse = z_q.reshape(1, -1)
    loss = coding_stage_loss(natural, coarse, z.reshape(1, -1), z_q.reshape(1, -1), eta=cfg.vq_commitment_eta)
    sr = int(data.get("sample_rate", 16000))
    bps = bitrate_bps(
        sample_rate=sr,
        temporal_downsample=cfg.temporal_downsample,
        frame_shift=cfg.frame_shift,
        codebook_size=cfg.codebook_size,
    )

    out = dict(data)
    out.update(
        fmelcodec_meta_block(sample_rate=sr),
    )
    out["fmelcodec"].update(
        {
            "coding_loss_proxy": round(float(loss), 4),
            "estimated_bitrate_bps": round(float(bps), 2),
            "code_index": int(_idx),
            "latent_mean_abs": round(float(np.abs(z).mean()), 6),
        }
    )
    return out
