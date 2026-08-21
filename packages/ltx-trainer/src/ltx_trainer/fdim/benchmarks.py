"""Reference metrics from FDIM paper tables (arXiv:2604.24123)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2604.24123"
PAPER_TITLE = (
    "FDIM: A Feature-distance-based Generic Video Quality Metric for Versatile Codecs"
)

# Table II — DCVQA codec-group (FDIM / FDIM Deep)
TABLE2_DCVQA_CODEC_GROUP: dict[str, dict[str, float]] = {
    "fdim_neural": {"plcc": 0.9010, "srocc": 0.8211},
    "fdim_traditional": {"plcc": 0.9312, "srocc": 0.8978},
    "fdim_deep_neural": {"plcc": 0.9012, "srocc": 0.8310},
    "fdim_deep_traditional": {"plcc": 0.9190, "srocc": 0.8898},
    "vmaf_neural": {"plcc": 0.8060, "srocc": 0.7304},
    "topiq_neural": {"plcc": 0.8514, "srocc": 0.7989},
}

# Table IV — five-dataset average (all-sequence)
TABLE4_SDR_AVERAGE: dict[str, dict[str, float]] = {
    "fdim": {"plcc": 0.9003, "srocc": 0.8771},
    "fdim_deep": {"plcc": 0.9007, "srocc": 0.8770},
    "vmaf": {"plcc": 0.8400, "srocc": 0.8177},
    "topiq": {"plcc": 0.8728, "srocc": 0.8596},
    "rankdvqa": {"plcc": 0.8228, "srocc": 0.8057},
}

# Table VII — 1080p 150-frame complexity (Deep branch)
TABLE7_COMPLEXITY_1080P: dict[str, dict[str, float]] = {
    "fdim_deep": {"params_m": 12.0, "flops_g": 211.59, "time_s": 13.79},
    "topiq": {"params_m": 33.27, "flops_g": 441.40, "time_s": 18.93},
    "rankdvqa": {"params_m": 4.59, "flops_g": 302.00, "time_s": 41.04},
    "lpips": {"params_m": 2.47, "flops_g": 59.48, "time_s": 1.55},
}

TRAINING_DCVQA = {
    "source_videos_train": 800,
    "distorted_sequences": 16136,
    "codecs": ["HEVC/x265", "AVS3", "EEM", "AlphaVC-P", "private NVC x2"],
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2_dcvqa_codec_group": TABLE2_DCVQA_CODEC_GROUP,
        "table4_sdr_average": TABLE4_SDR_AVERAGE,
        "table7_complexity_1080p": TABLE7_COMPLEXITY_1080P,
        "training_dcvqa": TRAINING_DCVQA,
        "github": "https://github.com/MCL-ZJU/FDIM",
    }
