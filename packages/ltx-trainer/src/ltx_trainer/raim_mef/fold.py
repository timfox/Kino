"""Fold RAIM MEF fusion readiness into HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def _stack_motion_proxy(stack: np.ndarray) -> float:
    """Mean frame-to-frame diff as alignment / motion proxy."""
    if stack.ndim < 2 or stack.shape[0] <= 1:
        return 0.0
    diffs = [float(np.abs(stack[i] - stack[i + 1]).mean()) for i in range(stack.shape[0] - 1)]
    return float(np.mean(diffs))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    ev_stack = data.get("hdr_ldr_ev_stack")
    ev_list = data.get("hdr_ev_list")
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}

    evs: list[float] = []
    if ev_list is not None:
        if hasattr(ev_list, "tolist"):
            raw = ev_list.tolist()
            evs = [float(x) for x in (raw if isinstance(raw, list) else [raw])]
        elif isinstance(ev_list, (list, tuple)):
            evs = [float(x) for x in ev_list]

    num_brackets = len(evs)
    motion = 0.0
    has_stack = ev_stack is not None
    if has_stack:
        arr = np.asarray(ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack)
        if arr.ndim >= 1:
            num_brackets = max(num_brackets, int(arr.shape[0]))
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 1 else arr.reshape(1, -1)
        motion = _stack_motion_proxy(flat)

    ev_span = float(max(evs) - min(evs)) if len(evs) >= 2 else 0.0
    bracket_score = float(np.clip(num_brackets / 5.0, 0.0, 1.0))
    dr_score = float(np.clip(ev_span / 8.0, 0.0, 1.0)) if ev_span > 0 else (0.6 if num_brackets >= 3 else 0.2)
    # Lower motion → higher fusion readiness (dynamic-scene MEF needs some motion but not extreme)
    motion_score = float(np.clip(1.0 - abs(motion - 0.08) / 0.25, 0.0, 1.0))
    readiness = float(np.clip(0.4 * bracket_score + 0.35 * dr_score + 0.25 * motion_score, 0.0, 1.0))

    out["raim_mef"] = {
        "arxiv_id": "2604.09030",
        "mef_fusion_readiness": round(readiness, 4),
        "num_brackets": int(num_brackets),
        "ev_span_stops": round(ev_span, 3),
        "motion_proxy": round(motion, 5),
        "has_ev_stack": has_stack,
        "track": hdr_meta.get("raim_track", "track2_mef"),
    }
    return out
