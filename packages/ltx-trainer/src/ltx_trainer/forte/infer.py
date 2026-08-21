"""Inference / caption env gates for FORTE prompt refinement (arXiv:2606.05812)."""

from __future__ import annotations

import os


def _enabled(name: str, *, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val not in ("0", "false", "no", "off")


def infer_forte_enabled() -> bool:
    """Stage-1 FOL refinement before live Gemma encode (``GOPEX_INFER_FORTE``; default on)."""
    return _enabled("GOPEX_INFER_FORTE")


def caption_forte_enabled() -> bool:
    """Refine captions during ``process_captions`` (``GOPEX_CAPTION_FORTE``; default on)."""
    return _enabled("GOPEX_CAPTION_FORTE")


def infer_forte_beam_online() -> bool:
    """Use online beam (B=3, D=2) instead of offline (B=5, D=4)."""
    return _enabled("GOPEX_INFER_FORTE_ONLINE", default="0")
