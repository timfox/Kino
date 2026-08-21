"""AV-fold sidecar: spatio-temporal / evolving-function proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dynamic_gp.constants import PAPER_ARXIV


def dynamic_gp_meta_block() -> dict[str, Any]:
    return {
        "dynamic_gp": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "spatiotemporal_function_estimation",
            "method": "dynamic_gaussian_process",
        }
    }


def _temporal_smoothness_proxy(arr: np.ndarray) -> float:
    """Frame-to-frame latent change proxy for evolving fields."""
    if arr.ndim < 4:
        return 0.5
    diffs = np.diff(arr.astype(np.float64), axis=0)
    return float(np.clip(1.0 / (1.0 + np.var(diffs)), 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg_block = dynamic_gp_meta_block()
    out = dict(data)
    out.update(cfg_block)

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("heat equation", "thermal", "diffusion", "spatiotemporal")):
        regime = "parabolic_pde"
    elif any(w in caption for w in ("wave equation", "propagating", "d alembert")):
        regime = "hyperbolic_pde"
    elif any(w in caption for w in ("kalman", "gaussian process", "dynamic gp", "ide")):
        regime = "dgp_estimation"
    elif any(w in caption for w in ("evolving function", "time-varying field", "integro-difference")):
        regime = "evolving_field"
    else:
        regime = "unknown_spatiotemporal"

    latents = data.get("latents")
    if latents is None:
        out["dynamic_gp"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    temporal = _temporal_smoothness_proxy(arr)
    spatial_std = float(arr.std())
    filter_readiness = float(np.clip(temporal * (0.4 + 0.6 * spatial_std), 0.0, 1.0))
    basis_fit = float(np.clip(1.0 - abs(spatial_std - 0.4), 0.0, 1.0))

    out["dynamic_gp"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "temporal_smoothness_proxy": round(temporal, 4),
            "basis_fit_proxy": round(basis_fit, 4),
            "filter_readiness_proxy": round(filter_readiness, 4),
            "default_basis_M": 31,
        }
    )
    return out
