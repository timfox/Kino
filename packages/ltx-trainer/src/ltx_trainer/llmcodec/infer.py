"""Env gates for LLMCodec vLLM weight-compression sidecars (arXiv:2606.05861)."""

from __future__ import annotations

import os


def _enabled(name: str, *, default: str = "1") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val not in ("0", "false", "no", "off")


def llmcodec_enabled() -> bool:
    """Enable LLMCodec lazy codec sidecars for served checkpoints (``GOPEX_LLMCODEC``)."""
    return _enabled("GOPEX_LLMCODEC")


def llmcodec_qp(default: int = 12) -> int:
    raw = os.environ.get("GOPEX_LLMCODEC_QP", str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default
