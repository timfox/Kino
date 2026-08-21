"""Best-of-N and EvoSearch selection (training-free quality boost)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.its_avgen.config import ItsAvgenConfig
from ltx_trainer.its_avgen.verifiers import combined_score, score_candidate


def adaptive_reweight(
    vr: float,
    js: float,
    *,
    vr_only_hack_risk: float = 0.0,
) -> tuple[float, float]:
    """ARW: reduce VR weight when verifier-hacking risk is high (e.g. spurious visual cues)."""
    risk = float(np.clip(vr_only_hack_risk, 0.0, 1.0))
    vr_w = 0.5 * (1.0 - 0.6 * risk)
    js_w = 1.0 - vr_w
    return vr_w, js_w


def best_of_n(
    candidates: list[dict[str, Any]],
    *,
    cfg: ItsAvgenConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or ItsAvgenConfig()
    if not candidates:
        raise ValueError("best_of_n requires at least one candidate")
    scored: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        vr_w, js_w = (cfg.vr_weight, cfg.javis_weight)
        if cfg.arw_enabled:
            vr_w, js_w = adaptive_reweight(
                float(cand.get("text_overlap", 0.5)),
                float(cand.get("av_sync", 0.5)),
                vr_only_hack_risk=float(cand.get("vr_hack_risk", 0.0)),
            )
        s = score_candidate(cand, seed=seed + i, vr_weight=vr_w, javis_weight=js_w)
        scored.append({"index": i, "scores": s, "candidate": cand})
    best = max(scored, key=lambda x: x["scores"]["combined"])
    return {"mode": "best_of_n", "n": len(candidates), "winner": best, "all": scored}


def evo_search_smoke(
    *,
    cfg: ItsAvgenConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Toy denoise-path search: perturb latent noise and keep improving combined score."""
    cfg = cfg or ItsAvgenConfig()
    rng = np.random.default_rng(seed)
    best_cand = {
        "text_overlap": 0.45,
        "motion_stability": 0.5,
        "av_sync": 0.42,
        "fine_grained_match": 0.4,
        "vr_hack_risk": 0.1,
    }
    best_score = score_candidate(best_cand, seed=seed, vr_weight=cfg.vr_weight, javis_weight=cfg.javis_weight)[
        "combined"
    ]
    trace: list[float] = [best_score]
    for step in range(cfg.evo_search_steps):
        trial = dict(best_cand)
        for k in trial:
            if isinstance(trial[k], float):
                trial[k] = float(np.clip(trial[k] + rng.normal(0, 0.05), 0.0, 1.0))
        s = score_candidate(trial, seed=seed + step + 1, vr_weight=cfg.vr_weight, javis_weight=cfg.javis_weight)[
            "combined"
        ]
        if s >= best_score:
            best_cand, best_score = trial, s
        trace.append(best_score)
    return {"mode": "evo_search", "steps": cfg.evo_search_steps, "best_score": best_score, "trace": trace}
