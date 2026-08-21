"""AV-fold sidecar: edge AR / multi-DNN scheduling hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.terastal.benchmarks import miss_rate_comparison_anchors
from ltx_trainer.terastal.constants import PAPER_ARXIV


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("ar ", "xr", "augmented reality", "edge", "real-time")):
        scenario = "ar_gaming_light"
    elif any(w in caption for w in ("multi-camera", "vision", "detection", "ssd")):
        scenario = "mcv_light"
    else:
        scenario = "generic_multi_dnn"
    out["terastal"] = {
        "arxiv_id": PAPER_ARXIV,
        "scenario_hint": scenario,
        "scheduler": "terastal",
        "use_layer_variants": "variant" in caption or "heterogeneous" in caption,
        "miss_rate_anchors": miss_rate_comparison_anchors(),
    }
    return out
