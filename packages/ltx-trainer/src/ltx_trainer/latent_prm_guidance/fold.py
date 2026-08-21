"""AV-fold sidecar: parallel code translation / latent reasoning hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.constants import PAPER_ARXIV, PRIMARY_MODEL


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("paratrans", "unipar", "parallax", "openmp", "cuda translation")):
        regime = "paratrans_translation"
    elif any(w in caption for w in ("latent prm", "latent reasoning", "process reward")):
        regime = "latent_prm_guidance"
    elif any(w in caption for w in ("hpc", "parallel code", "source-to-source")):
        regime = "hpc_code_translation"
    else:
        regime = "code_generation"
    out["latent_prm_guidance"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "primary_model": PRIMARY_MODEL,
        "intervention": "pre_decoding_latent_branch_selection",
        "benchmark": "ParaTrans",
    }
    return out
