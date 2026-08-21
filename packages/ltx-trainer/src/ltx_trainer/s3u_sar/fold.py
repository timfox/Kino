"""AV-fold sidecar: SAR / aircraft / scattering structure hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3u_sar.constants import PAPER_ARXIV


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("s3u-sar", "sar aircraft", "scattering keypoint", "semantic scattering")):
        regime = "s3u_sar"
    elif any(w in caption for w in ("sar", "synthetic aperture", "gaofen", "speckle")):
        regime = "sar_imagery"
    elif any(w in caption for w in ("aircraft", "airplane", "fuselage", "wing tip")):
        regime = "aircraft_structure"
    else:
        regime = "generic_remote_sensing"
    out["s3u_sar"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "representation": "semantic_scattering_structure",
    }
    return out
