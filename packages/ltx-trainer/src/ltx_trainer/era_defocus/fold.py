"""AV-fold sidecar: defocus severity + sharpness proxies for LTX training."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.era_defocus.constants import PAPER_ARXIV


def era_defocus_meta_block() -> dict[str, Any]:
    return {
        "era_defocus": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "defocus_deblur_proxy",
            "method": "error_aware_alm_unrolling",
        }
    }


def _latent_sharpness_proxy(arr: np.ndarray) -> float:
    """High-frequency energy proxy: frame-to-frame / spatial gradient variance."""
    if arr.ndim >= 3:
        # [T, C, H, W] or [C, H, W] — use spatial gradients on first slice
        frame = arr[0] if arr.ndim == 4 else arr
        if frame.ndim == 3:
            frame = frame.mean(axis=0)
    else:
        frame = arr
    gx = np.diff(frame.astype(np.float64), axis=-1)
    gy = np.diff(frame.astype(np.float64), axis=-2)
    grad_var = float(np.var(gx) + np.var(gy))
    return float(np.clip(grad_var * 10.0, 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg_block = era_defocus_meta_block()
    out = dict(data)
    out.update(cfg_block)

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("defocus", "bokeh", "depth of field", "dof blur", "era_defocus")):
        regime = "defocus_blur"
    elif any(w in caption for w in ("deblur", "sharp", "in focus", "tack sharp")):
        regime = "sharp_in_focus"
    elif any(w in caption for w in ("blur", "soft focus", "out of focus")):
        regime = "generic_blur"
    else:
        regime = "unknown_optics"

    latents = data.get("latents")
    if latents is None:
        out["era_defocus"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    sharpness = _latent_sharpness_proxy(arr)
    defocus_severity = float(np.clip(1.0 - sharpness, 0.0, 1.0))
    # Recoverable defocus: moderate severity with non-zero structure
    spatial_std = float(arr.std())
    deblur_readiness = float(np.clip(sharpness * (0.5 + 0.5 * spatial_std), 0.0, 1.0))
    mos_proxy = float(np.clip(1.0 + sharpness * 4.0, 1.0, 5.0))

    out["era_defocus"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "sharpness_proxy": round(sharpness, 4),
            "defocus_severity_proxy": round(defocus_severity, 4),
            "deblur_readiness_proxy": round(deblur_readiness, 4),
            "mos_proxy": round(mos_proxy, 3),
            "unrolling_depth": 10,
        }
    )
    return out
