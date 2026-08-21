"""Table 1–6 and summary anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.constants import (
    PARATRANS_TEST_TASKS,
    PRM_VS_LATENT_CI,
    PRM_VS_LATENT_GAIN_PP,
    PRM_VS_LATENT_P,
    TABLE1_VALIDATION,
    TABLE2_BRANCH_SELECTION,
    TABLE4_WLT,
    TABLE5_DIRECTION,
    TABLE6_ABLATION,
    TRAJECTORY_STATS,
)


def table_1_main() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_VALIDATION]


def table_2_branch_selection() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_BRANCH_SELECTION]


def table_4_wlt() -> dict[str, Any]:
    return dict(TABLE4_WLT)


def table_5_directions() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE5_DIRECTION]


def table_6_ablation() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE6_ABLATION]


def paired_significance() -> dict[str, Any]:
    return {
        "comparison": "Latent PRM guidance vs unguided latent reasoning",
        "gain_pp": dict(PRM_VS_LATENT_GAIN_PP),
        "ci_95": dict(PRM_VS_LATENT_CI),
        "p_value": dict(PRM_VS_LATENT_P),
        "unit": "task-level paired bootstrap / sign-flip over 76 tasks × 3 runs",
    }


def trajectory_analysis() -> dict[str, Any]:
    return dict(TRAJECTORY_STATS)


def summary_anchors() -> dict[str, Any]:
    prm = next(r for r in TABLE1_VALIDATION if r["method"] == "Latent PRM guidance")
    latent = next(r for r in TABLE1_VALIDATION if r["method"] == "Latent reasoning")
    return {
        "paratrans_test_tasks": PARATRANS_TEST_TASKS,
        "latent_reasoning_no_repair_pct": latent["no_repair"],
        "latent_prm_no_repair_pct": prm["no_repair"],
        "latent_prm_repair_pct": prm["repair"],
        "gain_pp_no_repair": PRM_VS_LATENT_GAIN_PP["no_repair"],
        "post_decode_text_prm_pct": 25.0,
        "post_decode_oracle_at_8_pct": 36.84,
        "filtering_interpretation": "PRM mainly filters harmful latent perturbations",
    }
