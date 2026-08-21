"""Fold LF-Diff bracket / tonemap readiness into HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import LF_DIFF_TONEMAP_MU_DEFAULT, lf_diff_meta_block


def _ev_span(evs: list[float] | tuple[float, ...] | None) -> float:
    if not evs:
        return 0.0
    vals = [float(x) for x in evs]
    return float(max(vals) - min(vals))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    mu = float(hdr_meta.get("tonemap_mu") or hdr_meta.get("lf_diff_mu") or LF_DIFF_TONEMAP_MU_DEFAULT)
    ev_list = data.get("hdr_ev_list")
    ev_stack = data.get("hdr_ldr_ev_stack")

    block = lf_diff_meta_block(mu=mu)
    out.update(block)

    evs: list[float] = []
    if ev_list is not None:
        if hasattr(ev_list, "tolist"):
            raw = ev_list.tolist()
            evs = [float(x) for x in (raw if isinstance(raw, list) else [raw])]
        elif isinstance(ev_list, (list, tuple)):
            evs = [float(x) for x in ev_list]

    num_brackets = len(evs) if evs else 0
    if ev_stack is not None:
        arr = np.asarray(ev_stack.detach().cpu().numpy() if hasattr(ev_stack, "detach") else ev_stack)
        num_brackets = max(num_brackets, int(arr.shape[0]) if arr.ndim >= 1 else 0)

    bracket_score = float(np.clip(num_brackets / 3.0, 0.0, 1.0))
    ev_span = _ev_span(evs)
    dr_score = float(np.clip(ev_span / 6.0, 0.0, 1.0)) if ev_span > 0 else (0.7 if num_brackets >= 3 else 0.25)
    mu_ok = 0.5 <= mu / LF_DIFF_TONEMAP_MU_DEFAULT <= 2.0
    readiness = float(np.clip(0.45 * bracket_score + 0.35 * dr_score + (0.2 if mu_ok else 0.0), 0.0, 1.0))

    out["lf_diff"].update(
        {
            "bracket_readiness": round(readiness, 4),
            "exposure_count": int(num_brackets),
            "ev_span_stops": round(ev_span, 3),
            "tonemap_mu": mu,
            "has_ev_stack": ev_stack is not None,
        }
    )
    return out
