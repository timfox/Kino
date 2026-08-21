"""Paper integration."""

from __future__ import annotations

from typing import Any

from ltx_trainer.legs_over_arms.benchmarks import benchmarks_bundle, table1_best_lower_body
from ltx_trainer.legs_over_arms.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    ref = table1_best_lower_body()
    b = benchmarks_bundle()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "backbone": "Human Scene Transformer (HST)",
        "task": "Multi-agent trajectory prediction from egocentric 360° robot perception",
        "method": {
            "features": "2D/3D skeletal keypoints + biomechanical cues G(s)",
            "finding": "Lower-body 3D keypoints K3D_L most predictive; 2D ERP keypoints still help",
            "temporal": "6 past @ 3Hz → 12 future (2s → 4s), 6 modes",
        },
        "results": {
            "JRDB_K3D_L_minADE": ref["minADE"],
            "JRDB_baseline_minADE": 0.39,
            "JRDB_ADE_reduction_pct": b["key_finding"]["K3D_L_minADE_reduction_pct"],
            "social_nav_K2D_reduction_pct": 7,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX stubs HST + skeletal configs, metrics, and Tables I–III. "
            "Datasets: JRDB + forthcoming GMU social-navigation ERP collection."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.legs_over_arms.mock import evaluation_smoke

    return evaluation_smoke()
