"""Fold LatentHDR exposure-readiness metadata into video / HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import latenthdr_meta_block


def _stack_spread_proxy(stack: np.ndarray) -> tuple[float, int]:
    """``stack`` ``[N,C,...]`` → mean inter-bracket latent diff, N."""
    if stack.ndim < 2:
        return 0.0, 1
    n = int(stack.shape[0])
    if n <= 1:
        return 0.0, n
    diffs = [float(np.abs(stack[i] - stack[i + 1]).mean()) for i in range(n - 1)]
    return float(np.mean(diffs)), n


def _ev_span(evs: list[float] | tuple[float, ...] | None) -> float:
    if not evs:
        return 0.0
    vals = [float(x) for x in evs]
    return float(max(vals) - min(vals))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    ev_stack = data.get("hdr_ldr_ev_stack")
    ev_list = data.get("hdr_ev_list")
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}

    recipe = hdr_meta.get("latenthdr", {}).get("synthetic_bracket_recipe", {})
    ev_min = float(recipe.get("ev_min", -7.0))
    ev_max = float(recipe.get("ev_max", 5.0))
    ev_step = float(recipe.get("ev_step", 1.0))
    expected_brackets = max(1, int(round((ev_max - ev_min) / ev_step)) + 1)

    evs: list[float] = []
    if ev_list is not None:
        if hasattr(ev_list, "tolist"):
            raw = ev_list.tolist()
            evs = [float(x) for x in (raw if isinstance(raw, list) else [raw])]
        elif isinstance(ev_list, (list, tuple)):
            evs = [float(x) for x in ev_list]

    block = latenthdr_meta_block(
        ev_recipe=(ev_min, ev_max, ev_step),
        save_ldr_stack=ev_stack is not None,
        ev_spec_saved=",".join(f"{e:g}" for e in evs[:8]) if evs else None,
    )
    out.update(block)

    num_brackets = len(evs) if evs else expected_brackets
    bracket_spread = 0.0
    has_stack = ev_stack is not None

    if has_stack:
        arr = np.asarray(
            ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack
        )
        bracket_spread, num_brackets = _stack_spread_proxy(arr)
        coverage = float(np.clip(num_brackets / max(expected_brackets, 1), 0.0, 1.0))
        spread_score = float(np.clip(bracket_spread * 4.0, 0.15, 1.0))
        readiness = float(np.clip(0.35 * coverage + 0.65 * spread_score, 0.0, 1.0))
    else:
        coverage = 0.0
        readiness = 0.25

    out["latenthdr"].update(
        {
            "exposure_readiness_proxy": round(readiness, 4),
            "bracket_spread_proxy": round(bracket_spread, 5),
            "bracket_coverage_proxy": round(coverage, 4),
            "num_brackets": int(num_brackets),
            "ev_span_stops": round(_ev_span(evs), 3),
            "has_ev_stack": has_stack,
            "expected_brackets": expected_brackets,
            "hdr_vae_encoding": hdr_meta.get("hdr_vae_encoding"),
        }
    )
    return out
