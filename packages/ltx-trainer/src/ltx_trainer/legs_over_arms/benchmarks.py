"""Reference metrics (Le et al. arXiv:2602.09076)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.legs_over_arms.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.legs_over_arms.datasets import datasets_bundle

# Table I — 3D skeletal configs on JRDB
TABLE1_JRDB_3D: list[dict[str, Any]] = [
    {"G": "∅ (baseline)", "minADE": 0.39, "minFDE": 0.64, "mLADE": 0.71, "NLLpos": 0.86},
    {"G": "K3D", "minADE": 0.37, "minFDE": 0.62, "mLADE": 0.68, "NLLpos": 0.88},
    {"G": "K3D ∪ C3D", "minADE": 0.36, "minFDE": 0.60, "mLADE": 0.62, "NLLpos": 0.80},
    {"G": "K3D_U", "minADE": 0.38, "minFDE": 0.58, "mLADE": 0.83, "NLLpos": 0.82},
    {"G": "K3D_U ∪ C3D_U", "minADE": 0.36, "minFDE": 0.57, "mLADE": 0.61, "NLLpos": 0.75},
    {"G": "K3D_L", "minADE": 0.34, "minFDE": 0.57, "mLADE": 0.61, "NLLpos": 0.65},
    {"G": "K3D_L ∪ C3D_L", "minADE": 0.34, "minFDE": 0.57, "mLADE": 0.69, "NLLpos": 0.67},
]

# Table II — 3D vs 2D on JRDB paired subset
TABLE2_JRDB_2D3D: list[dict[str, Any]] = [
    {"G": "∅ (baseline)", "minADE": 0.42, "minFDE": 0.62, "mLADE": 0.72, "NLLpos": 1.08},
    {"G": "K3D", "minADE": 0.39, "minFDE": 0.56, "mLADE": 0.65, "NLLpos": 0.88},
    {"G": "K3D_L", "minADE": 0.37, "minFDE": 0.54, "mLADE": 0.64, "NLLpos": 0.77},
    {"G": "K2D", "minADE": 0.41, "minFDE": 0.60, "mLADE": 0.74, "NLLpos": 1.08},
    {"G": "K2D_L", "minADE": 0.41, "minFDE": 0.59, "mLADE": 0.72, "NLLpos": 0.97},
]

# Table III — 2D keypoints from ERP on social-nav dataset
TABLE3_SOCIAL_NAV: list[dict[str, Any]] = [
    {"G": "∅ (baseline)", "minADE": 1.10, "minFDE": 1.44, "mLADE": 1.86, "NLLpos": 3.10},
    {"G": "K2D", "minADE": 1.02, "minFDE": 1.34, "mLADE": 1.60, "NLLpos": 2.80},
    {"G": "K2D_L", "minADE": 1.03, "minFDE": 1.35, "mLADE": 1.73, "NLLpos": 2.86},
]


def table1_best_lower_body() -> dict[str, Any]:
    return next(r for r in TABLE1_JRDB_3D if r["G"] == "K3D_L")


def ade_reduction_pct(baseline: float, improved: float) -> float:
    return (baseline - improved) / baseline * 100.0


def benchmarks_bundle() -> dict[str, Any]:
    best = table1_best_lower_body()
    base = next(r for r in TABLE1_JRDB_3D if "baseline" in r["G"])
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "datasets": datasets_bundle(),
        "table1_jrdb_3d": TABLE1_JRDB_3D,
        "table2_jrdb_2d3d": TABLE2_JRDB_2D3D,
        "table3_social_nav": TABLE3_SOCIAL_NAV,
        "key_finding": {
            "K3D_L_minADE_reduction_pct": round(
                ade_reduction_pct(base["minADE"], best["minADE"]), 1
            ),
            "social_nav_K2D_minADE_reduction_pct": 7.3,
        },
    }
