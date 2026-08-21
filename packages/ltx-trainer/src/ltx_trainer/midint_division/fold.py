"""AV-fold sidecar: exact arithmetic / crypto latent hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.constants import PAPER_ARXIV
from ltx_trainer.midint_division.cost_model import full_mult_bounds


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("crypto", "rsa", "bigint", "modular", "quotient")):
        regime = "crypto_midsize"
    elif any(w in caption for w in ("cuda", "cgbn", "gpu", "scan")):
        regime = "gpu_block_division"
    else:
        regime = "exact_integer"
    bounds = full_mult_bounds()
    out["midint_division"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "bits_exp_hint": 18 if "large" in caption else 15,
        "full_mult_cost_min": bounds["min"],
        "full_mult_cost_max": bounds["max"],
        "algorithm": "whole_shifted_inverse",
    }
    return out
