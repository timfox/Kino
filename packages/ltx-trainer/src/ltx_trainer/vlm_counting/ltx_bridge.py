"""LTX dataset QA: caption numerosity vs visual content."""

from __future__ import annotations

from typing import Any

import re

from ltx_trainer.vlm_counting.gaps import CountingDiagnostics, regime_for_n
from ltx_trainer.vlm_counting.probing import StonePresenceProbe


_NUM_RE = re.compile(r"\b(\d+)\b")


def extract_caption_count(caption: str) -> int | None:
    """Heuristic: first explicit integer in caption (e.g. 'three people' not handled)."""
    m = _NUM_RE.search(caption)
    return int(m.group(1)) if m else None


def caption_vision_gap_check(
    caption: str,
    probe: StonePresenceProbe,
    patch_embeddings,
    *,
    object_keyword: str | None = None,
) -> dict[str, Any]:
    """
    Compare caption-stated count vs hidden-number probe on clip/keyframe patches.
    Flags fractured-magnitude risk when language_gap is large but vision_gap is small.
    """
    ng_caption = extract_caption_count(caption)
    nh = float(probe.hidden_number(patch_embeddings.unsqueeze(0)).item())
    if ng_caption is None:
        return {"ok": False, "reason": "no_numeric_caption", "nh": round(nh, 2)}
    diag = CountingDiagnostics(ng=ng_caption, nh=nh, np_=ng_caption)
    return {
        "ok": diag.language_gap < 2.0 or diag.vision_gap > 2.0,
        "caption_ng": ng_caption,
        "hidden_nh": round(nh, 2),
        "vision_gap": round(diag.vision_gap, 2),
        "regime": regime_for_n(ng_caption),
        "object_keyword": object_keyword,
        "recommendation": "reject_for_training" if diag.vision_gap > 3 else "keep",
    }


def ltx_integration_notes() -> dict[str, Any]:
    return {
        "use_case": "Pre-merge QA on native captions with explicit counts",
        "pair_with": ["face_age_10k character consistency", "ltx_precomputed_audit"],
        "signal": "High vision_gap → bad caption; high language_gap with low vision_gap → VLM-style fracture",
    }
