"""Bernini-style semantic segment hints for Gemma vision captioning (arXiv:2605.22344)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bernini.fold import segment_ids_for_frames
from ltx_trainer.bernini.planner import inference_mask_ratio


def plan_clip_semantic_segments(
    duration_sec: float,
    *,
    num_frame_samples: int = 4,
    min_segments: int = 2,
    max_segments: int = 6,
) -> dict[str, Any]:
    """Plan coarse semantic segments for a single clip before VLM captioning."""
    dur = max(0.0, float(duration_sec))
    n_samples = max(1, int(num_frame_samples))
    # Match fold.py heuristics (latent frames unknown at caption time).
    latent_frames_guess = max(8, int(dur * 3.0))
    num_segments = min(max_segments, max(min_segments, min(6, max(2, latent_frames_guess // 8))))
    seg_ids = segment_ids_for_frames(n_samples, num_segments=num_segments)
    labels = _segment_labels(num_segments)
    frame_map = [
        {"frame_index": i, "segment_id": sid, "segment_label": labels[sid]}
        for i, sid in enumerate(seg_ids)
    ]
    plan_lines = [
        f"This clip is ~{dur:.1f}s. Treat it as {num_segments} semantic segment(s) for planning:",
    ]
    for idx, name in enumerate(labels):
        plan_lines.append(f"  segment {idx} ({name})")
    plan_lines.append(
        "Describe motion and subject changes at segment boundaries when visible; "
        "keep one coherent paragraph in the caption JSON."
    )
    return {
        "num_segments": num_segments,
        "segment_ids": seg_ids,
        "segment_labels": labels,
        "frame_segment_map": frame_map,
        "inference_mask_ratio_k0": round(inference_mask_ratio(0, 8), 4),
        "planner_prompt": "\n".join(plan_lines),
    }


def _segment_labels(num_segments: int) -> list[str]:
    if num_segments <= 1:
        return ["full_clip"]
    if num_segments == 2:
        return ["opening", "closing"]
    if num_segments == 3:
        return ["opening", "middle", "closing"]
    return [f"part_{i + 1}" for i in range(num_segments)]


def attach_bernini_caption_meta(meta: dict[str, Any], duration_sec: float, *, num_frame_samples: int) -> dict[str, Any]:
    """Copy ``meta`` and add ``bernini`` block when enabled via env (default on)."""
    import os

    if os.environ.get("GOPEX_BERNINI_CAPTION_HINTS", "1").strip().lower() in ("0", "false", "no"):
        return meta
    out = dict(meta)
    plan = plan_clip_semantic_segments(duration_sec, num_frame_samples=num_frame_samples)
    out["bernini"] = plan
    out["bernini_num_segments"] = plan["num_segments"]
    return out


def bernini_user_prompt_lines(meta: dict[str, Any] | None) -> list[str]:
    if not meta:
        return []
    block = meta.get("bernini")
    if not isinstance(block, dict):
        return []
    prompt = block.get("planner_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return []
    return [
        "--- Bernini semantic segment plan (structure only; trust pixels) ---",
        prompt.strip(),
        "--- End Bernini plan ---",
    ]


def append_bernini_caption_suffix(caption: str, meta: dict[str, Any] | None) -> str:
    """Light suffix so segment count is recoverable from ``dataset.json`` text."""
    if not meta:
        return caption
    n = meta.get("bernini_num_segments")
    if n is None:
        block = meta.get("bernini")
        if isinstance(block, dict):
            n = block.get("num_segments")
    if not isinstance(n, int) or n < 2:
        return caption
    suffix = f" Semantic segments: {n}."
    if suffix.strip().lower() in caption.lower():
        return caption
    return f"{caption.rstrip()}{suffix}"
