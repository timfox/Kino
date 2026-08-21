"""Augment Engineering smoke (arXiv:2605.26146)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.augment_engineering.config import AugmentEngConfig
from ltx_trainer.augment_engineering.metrics import (
    coverage_breadth,
    orchestration_overhead,
    transfer_velocity_hours,
)
from ltx_trainer.augment_engineering.phases import orchestration_patterns, phase_catalog
from ltx_trainer.augment_engineering.rubric import classify_prompt_level
from ltx_trainer.augment_engineering.stats import cochran_armitage_trend, wrights_law_fit
from ltx_trainer.augment_engineering.tables import headline_results, longitudinal_phases, table6_pipeline_overhead


def evaluation_smoke(cfg: AugmentEngConfig | None = None) -> dict[str, Any]:
    c = cfg or AugmentEngConfig()
    qd = {
        "software_development": 1,
        "academic_publication": 0,  # mixed peer review — paper does not claim uniform Qd=1
        "proposals": 1,
        "curriculum_design": 1,
        "video_production": 1,
        "presentation_design": 1,
        "web_deployment": 1,
    }
    cb = coverage_breadth(qd)
    tv_n3 = transfer_velocity_hours(168.0, 3, beta=0.5)
    oh_train = orchestration_overhead(0.18, 0.82)

    level = classify_prompt_level(
        context_files=5,
        has_schema=True,
        automated=False,
        has_role_or_structure=True,
    )
    trend = cochran_armitage_trend()
    wl = wrights_law_fit([30.0, 12.0, 9.0, 8.0, 5.0], paper_beta=0.44)
    headlines = headline_results()

    return {
        "paper": c.paper_arxiv,
        "coverage_breadth_cb": cb,
        "transfer_velocity_n3_hours": round(tv_n3, 2),
        "orchestration_overhead_training_pipeline": round(oh_train, 3),
        "prompt_level_toy": level,
        "cochran_z": trend.z_statistic,
        "cochran_p_approx": trend.p_value_approx,
        "first_pass_rates_by_level": trend.rates,
        "wright_beta_hat": wl.beta,
        "wright_n_artifacts_paper": headlines["wright_law_n_artifacts"],
        "orchestration_phases": len(phase_catalog()),
        "orchestration_patterns": len(orchestration_patterns()),
        "ai_tools_in_portfolio": len(c.ai_tools),
        "longitudinal_phase3_coverage": longitudinal_phases()[-1]["coverage_breadth"],
        "table6_max_oh_pct": max(r["oh_pct"] for r in table6_pipeline_overhead()),
        "paper_production_loc": headlines["production_code_loc"],
    }
