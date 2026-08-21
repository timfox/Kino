"""Inference env gates for SVHighlights recap / saliency (arXiv:2606.06926)."""

from __future__ import annotations

import os


def _enabled(name: str, *, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val not in ("0", "false", "no", "off")


def svhighlights_fold_enabled() -> bool:
    """Write TF-SELECTOR saliency sidecars at VAE encode (``GOPEX_SVHIGHLIGHTS_ENABLE``)."""
    return _enabled("GOPEX_SVHIGHLIGHTS_ENABLE")


def infer_svhighlights_enabled() -> bool:
    """Prefer top salient spans at delivery / recap inference (``GOPEX_INFER_SVHIGHLIGHTS``)."""
    return _enabled("GOPEX_INFER_SVHIGHLIGHTS")


def svhighlights_top_fraction() -> float:
    raw = os.environ.get("GOPEX_SVHIGHLIGHTS_TOP_FRACTION", "0.15").strip()
    try:
        val = float(raw)
    except ValueError:
        val = 0.15
    return max(0.01, min(1.0, val))
