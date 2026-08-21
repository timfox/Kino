"""FlatSounds framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.flatsounds.alignment import hit_coverage, perfect_align, timing_error_ms
from ltx_trainer.flatsounds.benchmarks import benchmarks_bundle
from ltx_trainer.flatsounds.confidence import (
    pair_direction_vote,
    quality_weight,
    weighted_confidence,
)
from ltx_trainer.flatsounds.config import FlatSoundsConfig
from ltx_trainer.flatsounds.physics_metrics import METRIC_FN


def framework_card(cfg: FlatSoundsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FlatSoundsConfig()
    return {
        "name": "FlatSounds",
        "paper": cfg.paper_arxiv,
        "title": "Benchmarking Single-Factor Physical Video-to-Audio Generation",
        "dataset_clips": cfg.dataset_clips,
        "physics_test_cases": cfg.physics_test_cases,
        "modes": ["counterfactual_pairs", "single_video_patterns"],
        "metrics": list(METRIC_FN.keys()) + ["hit_coverage", "timing_error_ms", "perfect_align"],
    }


def knowledge_card(cfg: FlatSoundsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FlatSoundsConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_findings": {
            "text_improves_physics_hurts_timing": bench["caption_hurts_timing"],
            "confidence_correlates_human": bench["confidence_beats_desync_spearman"],
            "top_elo": max(bench["table7_elo"].items(), key=lambda x: x[1])[0],
        },
        "integration": {
            "env": "GOPEX_FLATSOUNDS=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,flatsounds",
            "cli": "./scripts/gopex-flatsounds.sh",
        },
    }


def evaluation_demo(cfg: FlatSoundsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FlatSoundsConfig()
    sr = cfg.analysis_sr_hz
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # Synthetic impacts at 0.5s and 1.2s
    audio = np.zeros_like(t)
    for onset in (0.5, 1.2):
        idx = int(onset * sr)
        burst = rng.standard_normal(min(800, len(t) - idx)) * np.exp(
            -np.linspace(0, 4, min(800, len(t) - idx))
        )
        audio[idx : idx + len(burst)] += burst

    ann = np.array([0.5, 1.2])
    cov = hit_coverage(audio, ann, sr=sr)
    terr = timing_error_ms(audio, ann, sr=sr)

    # Pair: B brighter / harder → higher centroid
    y_a = audio
    y_b = audio * 1.2 + 0.05 * rng.standard_normal(len(audio))
    m_a = METRIC_FN["spectral_centroid"](y_a, sr)
    m_b = METRIC_FN["spectral_centroid"](y_b, sr)
    vote = pair_direction_vote(m_a, m_b, "increase")
    w = quality_weight(cov, cov, 0.7, 0.7)
    conf = weighted_confidence([vote], [w])

    seeds_cov = [cov, cov * 0.95, 1.0, 0.88]
    palign = perfect_align(seeds_cov)

    return {
        "hit_coverage": round(cov, 4),
        "timing_error_ms": round(terr, 2) if np.isfinite(terr) else None,
        "perfect_align_pct": round(palign, 2),
        "spectral_centroid_pair_vote": vote,
        "weighted_confidence_stub": round(conf, 4),
        "paper_tables": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: FlatSoundsConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    bench = benchmarks_bundle()
    top = max(bench["table2_overall"].items(), key=lambda x: x[1]["confidence"])
    return {
        "ok": demo["hit_coverage"] > 0,
        "hit_coverage": demo["hit_coverage"],
        "confidence_leader": top[0],
        "confidence_leader_score": top[1]["confidence"],
        "caption_hurts_timing": bench["caption_hurts_timing"],
        "physics_test_cases": bench["physics_test_cases"],
    }
