"""Query-agnostic CAPC-style text compression (CPU demo, not LLMLingua)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.capc.config import CapcConfig
from ltx_trainer.capc.cost_model import tier_preserving_rmax, tier_preserving_rmax_chars


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def estimate_tokens(text: str, cfg: CapcConfig | None = None) -> int:
    c = cfg or CapcConfig()
    return max(1, int(round(len(text) / max(1e-6, c.chars_per_token))))


def compress_query_agnostic(
    text: str,
    *,
    ratio: float | None = None,
    max_chars: int | None = None,
    cfg: CapcConfig | None = None,
) -> dict[str, Any]:
    """Compress ``text`` without conditioning on a query.

    Prefers sentence-boundary truncation so the retained prefix stays stable
    across turns (CAPC cache-friendly property). When ``max_chars`` is set,
    ratio is chosen to fit that budget (wake path).
    """
    c = cfg or CapcConfig()
    raw = (text or "").strip()
    if not raw:
        return {
            "text": "",
            "original_chars": 0,
            "compressed_chars": 0,
            "ratio": 1.0,
            "truncated": False,
            "method": "empty",
        }

    orig_chars = len(raw)
    orig_tok = estimate_tokens(raw, c)

    if max_chars is not None and max_chars > 0:
        r_eff = float(tier_preserving_rmax_chars(orig_chars, max_chars, c))
        target_chars = max_chars
    else:
        r = float(ratio) if ratio is not None else float(tier_preserving_rmax(orig_tok, c))
        r_eff = max(1.0, r)
        target_chars = max(1, int(round(orig_chars / r_eff)))

    if orig_chars <= target_chars:
        return {
            "text": raw,
            "original_chars": orig_chars,
            "compressed_chars": orig_chars,
            "original_tokens_est": orig_tok,
            "ratio": 1.0,
            "truncated": False,
            "method": "passthrough",
        }

    parts = [p.strip() for p in _SENTENCE_SPLIT.split(raw) if p and p.strip()]
    if len(parts) <= 1:
        out = raw[:target_chars].rstrip()
        if len(out) < orig_chars:
            out = out + "\n…[capc truncated]"
        return {
            "text": out,
            "original_chars": orig_chars,
            "compressed_chars": len(out),
            "original_tokens_est": orig_tok,
            "ratio": round(orig_chars / max(1, len(out)), 3),
            "truncated": True,
            "method": "char_slice",
        }

    kept: list[str] = []
    size = 0
    for sent in parts:
        add = len(sent) + (1 if kept else 0)
        if size + add > target_chars and kept:
            break
        kept.append(sent)
        size += add
    if not kept:
        kept = [parts[0][:target_chars]]

    out = " ".join(kept)
    if len(out) > target_chars:
        out = out[:target_chars].rstrip()
    if len(out) < orig_chars:
        out = out.rstrip() + "\n…[capc truncated]"

    return {
        "text": out,
        "original_chars": orig_chars,
        "compressed_chars": len(out),
        "original_tokens_est": orig_tok,
        "ratio": round(orig_chars / max(1, len(out)), 3),
        "truncated": True,
        "method": "sentence_prefix",
        "r_planned": r_eff,
    }
