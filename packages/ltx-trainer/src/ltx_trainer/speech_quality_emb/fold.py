"""Fold speech-quality embedding diagnostics into LTX audio shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.speech_quality_emb.config import SpeechQualityEmbConfig
from ltx_trainer.speech_quality_emb.detection import cosine_similarity_to_enrollment
from ltx_trainer.speech_quality_emb.mixup import partial_mixup_waveform_mask, pseudo_frame_scores


def speech_quality_meta_block() -> dict[str, Any]:
    return {
        "speech_quality_emb": {
            "arxiv_id": "2605.21332",
            "fold_role": "audio_quality_sidecar",
            "method": "LSSQA-style embedding similarity (proxy from audio latents)",
        }
    }


def _frame_embeddings_from_latents(latents: Any, *, frames: int, dim: int) -> np.ndarray:
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return np.zeros((frames, dim), dtype=np.float64)
    step = max(1, flat.size // frames)
    emb = []
    for f in range(frames):
        chunk = flat[f * step : (f + 1) * step]
        if chunk.size < dim:
            chunk = np.pad(chunk, (0, dim - chunk.size))
        v = chunk[:dim].astype(np.float64)
        emb.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(emb, axis=0)


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = SpeechQualityEmbConfig()
    latents = data.get("latents")
    out = dict(data)
    out.update(speech_quality_meta_block())
    if latents is None:
        return out

    frames = max(1, int(data.get("num_time_steps", 8)))
    emb = _frame_embeddings_from_latents(latents, frames=frames, dim=cfg.embedding_dim)
    enroll = emb.mean(axis=0)
    sim = cosine_similarity_to_enrollment(emb, enroll)
    threshold = float(cfg.degraded_similarity_threshold)
    degraded = (sim < threshold).astype(np.int64)

    rng = np.random.default_rng(7)
    mq = partial_mixup_waveform_mask(
        frames,
        rng=rng,
        min_seg_len=1,
        max_seg_len=max(1, min(50, frames)),
    )
    q_ref = 0.9 + 0.05 * rng.standard_normal(frames)
    q_deg = 0.4 + 0.1 * rng.standard_normal(frames)
    pseudo = pseudo_frame_scores(q_ref, q_deg, mq)

    out["speech_quality_emb"].update(
        {
            "mean_cosine_sim": round(float(sim.mean()), 4),
            "degraded_frame_ratio": round(float(degraded.mean()), 4),
            "pseudo_mos_mean": round(float(pseudo.mean()), 4),
            "frames": frames,
        }
    )
    return out
