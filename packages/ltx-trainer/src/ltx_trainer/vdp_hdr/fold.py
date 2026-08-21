"""Fold VDP-HDR bracket fusion readiness into video / HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import vdp_hdr_meta_block


def _bracket_stats(stack: np.ndarray) -> tuple[float, int]:
    if stack.ndim < 1:
        return 0.0, 0
    n = int(stack.shape[0])
    if n <= 1:
        return 0.0, n
    diffs = [float(np.abs(stack[i] - stack[i + 1]).mean()) for i in range(n - 1)]
    return float(np.mean(diffs)), n


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    num_frames = int(hdr_meta.get("vdp_bracket_frames") or hdr_meta.get("bracket_frames") or 5)
    block = vdp_hdr_meta_block(num_frames=num_frames)
    out.update(block)

    ev_stack = data.get("hdr_ldr_ev_stack")
    if ev_stack is None:
        ev_stack = data.get("ldr_bracket")
    ev_list = data.get("hdr_ev_list")
    spread, n = 0.0, num_frames
    if ev_stack is not None:
        arr = np.asarray(ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack)
        if arr.ndim >= 1:
            spread, n = _bracket_stats(arr)
    elif ev_list is not None:
        n = len(ev_list) if hasattr(ev_list, "__len__") else num_frames

    coverage = float(np.clip(n / max(num_frames, 1), 0.0, 1.0))
    ev_span = 0.0
    if ev_list is not None and hasattr(ev_list, "__len__") and len(ev_list) >= 2:
        evs = [float(x) for x in ev_list]
        ev_span = float(max(evs) - min(evs))
    dr_score = float(np.clip(ev_span / 6.0, 0.0, 1.0)) if ev_span > 0 else coverage * 0.7
    motion_score = float(np.clip(1.0 - abs(spread - 0.12) / 0.3, 0.0, 1.0))
    readiness = float(np.clip(0.4 * coverage + 0.35 * dr_score + 0.25 * motion_score, 0.0, 1.0))

    out["vdp_hdr"].update(
        {
            "fusion_readiness_proxy": round(readiness, 4),
            "bracket_spread_proxy": round(spread, 5),
            "num_brackets": int(n),
            "ev_span_stops": round(ev_span, 3),
            "has_ev_stack": ev_stack is not None,
        }
    )
    return out
