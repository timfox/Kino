"""Table 1 and summary anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.era_defocus.constants import (
    BEST_BASELINE_PSNR,
    DATASET_DPDD,
    ERA_ROW,
    ERA_WO_E_ROW,
    TABLE1_RESULTS,
)


def table_1_results() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_RESULTS]


def dataset_card() -> dict[str, Any]:
    return dict(DATASET_DPDD)


def summary_anchors() -> dict[str, Any]:
    era_psnr = float(ERA_ROW["DPDD_PSNR"])
    return {
        "DPDD_PSNR": era_psnr,
        "DPDD_SSIM": float(ERA_ROW["DPDD_SSIM"]),
        "RealDOF_PSNR": float(ERA_ROW["RealDOF_PSNR"]),
        "RTF_PSNR": float(ERA_ROW["RTF_PSNR"]),
        "gain_vs_best_baseline_db": round(era_psnr - BEST_BASELINE_PSNR, 3),
        "error_term_gain_db": round(era_psnr - float(ERA_WO_E_ROW["DPDD_PSNR"]), 3),
        "unrolling_depth": 10,
        "kernel_size": 61,
    }
