"""AV-fold sidecar: registration / change-detection proxies for LTX training."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dilated_sym_diff.constants import PAPER_ARXIV


def dilated_sym_diff_meta_block() -> dict[str, Any]:
    return {
        "dilated_sym_diff": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "binary_change_detection_proxy",
            "method": "dilated_symmetric_difference",
        }
    }


def _boundary_contrast_proxy(arr: np.ndarray) -> float:
    """High spatial gradient → sharp binary-like boundaries."""
    if arr.ndim >= 3:
        frame = arr[0] if arr.ndim == 4 else arr
        if frame.ndim == 3:
            frame = frame.mean(axis=0)
    else:
        frame = arr
    gx = np.diff(frame.astype(np.float64), axis=-1)
    gy = np.diff(frame.astype(np.float64), axis=-2)
    return float(np.clip((np.var(gx) + np.var(gy)) * 5.0, 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg_block = dilated_sym_diff_meta_block()
    out = dict(data)
    out.update(cfg_block)

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("change detection", "before after", "symmetric difference", "binary compare")):
        regime = "change_detection"
    elif any(w in caption for w in ("registration", "misalign", "alignment error", "registered")):
        regime = "registration"
    elif any(w in caption for w in ("morphology", "dilation", "binary mask", "segmentation diff")):
        regime = "morphology"
    elif any(w in caption for w in ("additive manufacturing", "quality control", "aerial compare")):
        regime = "qc_inspection"
    else:
        regime = "unknown_comparison"

    latents = data.get("latents")
    if latents is None:
        out["dilated_sym_diff"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    boundary = _boundary_contrast_proxy(arr)
    alignment_proxy = float(np.clip(1.0 - boundary * 0.3, 0.0, 1.0))
    change_readiness = float(np.clip(boundary * (0.5 + 0.5 * float(arr.std())), 0.0, 1.0))

    out["dilated_sym_diff"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "boundary_contrast_proxy": round(boundary, 4),
            "alignment_error_proxy": round(alignment_proxy, 4),
            "change_detection_readiness_proxy": round(change_readiness, 4),
            "default_dilation_r": 8,
        }
    )
    return out
