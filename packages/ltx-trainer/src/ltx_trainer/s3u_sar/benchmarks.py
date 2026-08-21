"""Table I–IV and summary anchors (Sec. IV)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3u_sar.constants import (
    DATASET_KP_SAR,
    ORIENTATION_GAIN_P1,
    ORIENTATION_GAIN_P5,
    TABLE1_BASELINES_AP,
    TABLE1_S3U_HRNET_W32,
    TABLE2_ABLATION,
    TABLE3_CROSS_CATEGORY,
    TABLE4_ORIENTATION,
)


def table_1_baselines() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE1_BASELINES_AP]


def table_1_s3u_metrics() -> dict[str, float]:
    return dict(TABLE1_S3U_HRNET_W32)


def table_2_ablation() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_ABLATION]


def table_3_cross_category() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE3_CROSS_CATEGORY]


def table_4_orientation() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE4_ORIENTATION]


def dataset_card() -> dict[str, Any]:
    return dict(DATASET_KP_SAR)


def summary_anchors() -> dict[str, Any]:
    s3u = TABLE1_S3U_HRNET_W32
    hrnet_w32 = next(r for r in TABLE1_BASELINES_AP if r["method"] == "HRNet-W32")
    return {
        "AP": s3u["AP"],
        "AP_improvement_vs_hrnet_w32": round(s3u["AP"] - float(hrnet_w32["AP"]), 1),
        "AP75": s3u["AP75"],
        "orientation_P1": 30.13,
        "orientation_P5": 68.53,
        "orientation_MAE_deg": 13.15,
        "P1_gain_pp": ORIENTATION_GAIN_P1,
        "P5_gain_pp": ORIENTATION_GAIN_P5,
        "dataset_samples": DATASET_KP_SAR["samples"],
        "num_keypoints": 10,
    }
