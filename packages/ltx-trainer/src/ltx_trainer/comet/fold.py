"""Fold COMET PLSHead proxies into LTX ``audio_latents`` / condition sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.pls import pls_svd
from ltx_trainer.comet.plshead import head_energy_ratio, plshead_truncate


def comet_meta_block() -> dict[str, Any]:
    cfg = CometConfig()
    return {
        "comet": {
            "arxiv_id": "2605.29628",
            "embed_dim": cfg.embed_dim,
            "head_size": cfg.head_size,
            "method": "PLSHead",
            "fold_role": "clap_embedding_sidecar",
        }
    }


def _as_vector(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x.detach().cpu().float().numpy() if hasattr(x, "detach") else x, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        return None
    return arr


def _pick_vector(data: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key not in data or data[key] is None:
            continue
        vec = _as_vector(data[key])
        if vec is not None:
            return vec
    return None


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach COMET head-energy proxy when a CLAP-like embedding is present."""
    out = dict(data)
    out.update(comet_meta_block())

    cfg = CometConfig()
    text_emb = _pick_vector(data, ("clap_text_embedding", "text_embedding"))
    audio_emb = _pick_vector(data, ("clap_audio_embedding", "audio_embedding"))

    if text_emb is not None and audio_emb is not None and text_emb.size == audio_emb.size:
        decomp = pls_svd(text_emb.reshape(1, -1), audio_emb.reshape(1, -1))
        t100 = plshead_truncate(text_emb, mean=decomp["t_mean"], directions=decomp["U"], head_size=cfg.head_size)
        a100 = plshead_truncate(audio_emb, mean=decomp["a_mean"], directions=decomp["V"], head_size=cfg.head_size)
        energy = head_energy_ratio(t100, head_size=min(cfg.head_size, t100.size))
        out["comet"].update(
            {
                "head_energy_fraction": round(energy["head_fraction"], 4),
                "modality_gap_mitigation": "PLSHead",
                "truncated_dim": int(t100.size),
                "pls_svd": True,
            }
        )
        return out

    emb = text_emb or audio_emb
    if emb is None:
        lat = data.get("latents")
        if lat is not None:
            flat = np.asarray(lat, dtype=np.float64).ravel()
            if flat.size >= CometConfig().embed_dim:
                emb = flat[: CometConfig().embed_dim]
    if emb is None:
        return out

    # Fallback: identity-basis head-energy proxy
    coef = emb / (np.linalg.norm(emb) + 1e-12)
    energy = head_energy_ratio(coef, head_size=min(cfg.head_size, coef.size))
    out["comet"].update(
        {
            "head_energy_fraction": round(energy["head_fraction"], 4),
            "modality_gap_mitigation": "PLSHead-ready",
            "truncated_dim": min(cfg.head_size, coef.size),
            "pls_svd": False,
        }
    )
    return out
