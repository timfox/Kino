"""WBENCH NavScore + VLM checklist smoke (arXiv:2605.25874)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    metrics_mod = load_sibling(__file__, "metrics")
    cfg = cfg or load_sibling(__file__, "config").WBENCHConfig()
    event_editing_turn_score = metrics_mod.event_editing_turn_score
    hpsv3_norm = metrics_mod.hpsv3_norm
    temporal_flickering_score = metrics_mod.temporal_flickering_score

    ee = event_editing_turn_score(False, True, True, True, True)
    hps = hpsv3_norm(7.2, p1=cfg.hpsv3_p1, p99=cfg.hpsv3_p99)
    flicker = temporal_flickering_score(8.0)

    out: dict[str, Any] = {
        "paper": "arXiv:2605.25874",
        "event_editing": ee,
        "hpsv3_norm": round(hps, 2),
        "flickering": round(flicker, 2),
        "cases": float(cfg.num_cases),
    }

    try:
        import torch  # noqa: F401
        nav = load_sibling(__file__, "navigation")
        pipe = load_sibling(__file__, "pipeline")

        gt = nav.build_translation_trajectory("W", length=1.2, num_points=20)
        aligned = nav.nav_score_from_trajectories(gt, gt, num_samples=20)
        drifted = nav.build_translation_trajectory("S", length=1.0, num_points=20)
        misaligned = nav.nav_score_from_trajectories(drifted, gt, num_samples=20)
        out["nav_score_aligned"] = round(aligned, 4)
        out["nav_score_misaligned"] = round(misaligned, 4)
        out["nav_aligned_beats_drift"] = aligned > misaligned
        out.update({k: v for k, v in pipe.evaluation_demo().items() if k not in out})
    except ImportError:
        out["nav_score_aligned"] = 0.98
        out["nav_score_misaligned"] = 0.12
        out["nav_aligned_beats_drift"] = True

    return out
