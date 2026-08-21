"""Human-alignment pilot study (Sec. 4.5, Fig. 10)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.config import LongAVCompassConfig
from ltx_trainer.longav_compass.scoring import pearson_r


def table_human_alignment_pearson(cfg: LongAVCompassConfig | None = None) -> dict[str, float]:
    """Reported Pearson ρ between human and benchmark pairwise win rates."""
    cfg = cfg or LongAVCompassConfig()
    return {
        "content_fidelity": cfg.human_alignment_pearson[0],
        "visual_quality": cfg.human_alignment_pearson[1],
        "long_video_stability": cfg.human_alignment_pearson[2],
    }


def pilot_model_win_rates() -> list[dict[str, Any]]:
    """Excerpt of model-level win rates on 40-case pilot (synthetic, rank-preserving)."""
    return [
        {
            "model": "Seedance 2.0",
            "human": {"content_fidelity": 0.78, "visual_quality": 0.82, "long_video_stability": 0.80},
            "benchmark": {"content_fidelity": 0.76, "visual_quality": 0.84, "long_video_stability": 0.79},
        },
        {
            "model": "Kling 3.0",
            "human": {"content_fidelity": 0.81, "visual_quality": 0.76, "long_video_stability": 0.83},
            "benchmark": {"content_fidelity": 0.79, "visual_quality": 0.78, "long_video_stability": 0.81},
        },
        {
            "model": "Veo 3.1",
            "human": {"content_fidelity": 0.62, "visual_quality": 0.68, "long_video_stability": 0.58},
            "benchmark": {"content_fidelity": 0.60, "visual_quality": 0.66, "long_video_stability": 0.57},
        },
        {
            "model": "LTX 2.3",
            "human": {"content_fidelity": 0.52, "visual_quality": 0.48, "long_video_stability": 0.50},
            "benchmark": {"content_fidelity": 0.50, "visual_quality": 0.46, "long_video_stability": 0.49},
        },
        {
            "model": "Open-Sora",
            "human": {"content_fidelity": 0.28, "visual_quality": 0.25, "long_video_stability": 0.30},
            "benchmark": {"content_fidelity": 0.26, "visual_quality": 0.24, "long_video_stability": 0.29},
        },
    ]


def human_alignment_validation(cfg: LongAVCompassConfig | None = None) -> dict[str, Any]:
    """Compute Pearson ρ on pilot excerpt; compare to paper constants."""
    cfg = cfg or LongAVCompassConfig()
    rows = pilot_model_win_rates()
    paper = table_human_alignment_pearson(cfg)
    computed: dict[str, float] = {}
    for dim in ("content_fidelity", "visual_quality", "long_video_stability"):
        human = [r["human"][dim] for r in rows]
        bench = [r["benchmark"][dim] for r in rows]
        computed[dim] = round(pearson_r(human, bench), 4)
    return {
        "pilot_cases": 40,
        "dimensions": {
            "content_fidelity": "VQA + TVAlign aggregate",
            "visual_quality": "VQ + Hol. aggregate",
            "long_video_stability": "Cont. + Trans. aggregate",
        },
        "paper_pearson": paper,
        "computed_pearson_pilot_excerpt": computed,
        "models": rows,
    }
