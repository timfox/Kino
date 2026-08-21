"""Bernini planner smoke."""

from __future__ import annotations

from typing import Any


def toy_segment_ids(num_tokens: int, *, num_segments: int = 3) -> list[int]:
    return [i % num_segments for i in range(num_tokens)]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.bernini.planner import inference_mask_ratio, visible_token_fraction

    segs = toy_segment_ids(24, num_segments=3)
    mr = inference_mask_ratio(2, 10)
    return {
        "unique_segments": len(set(segs)),
        "inference_mask_ratio": round(mr, 4),
        "visible_fraction": round(visible_token_fraction(mr), 4),
    }
