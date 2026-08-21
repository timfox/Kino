"""Reference metrics from PhysHDR-GS (arXiv:2603.28020)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.physthdr_gs.config import EXPOSURE_TIMES, ExposureSetting
from ltx_trainer.physthdr_gs.dataset import DATASETS
from ltx_trainer.physthdr_gs.metrics import (
    PSNR_GAIN_OVER_HDR_GS,
    TABLE1_EXP3_HDR_GS,
    TABLE1_EXP3_OURS_DAGGER,
    TABLE2_EXP3,
    TABLE3_EFFICIENCY,
    TABLE4_ABLATION,
)

PAPER_ARXIV = "2603.28020"
PAPER_TITLE = "Physically Inspired Gaussian Splatting for HDR Novel View Synthesis"

TRAINING_DEFAULTS = {
    "iterations": 30_000,
    "freeze_fmix_iters": 10_000,
    "lambda_rec": 1.0,
    "lambda_cons": 0.5,
    "lambda_unit_synthetic": 0.5,
    "gamma_mse": 0.2,
    "scale_s_igs": 1.0,
    "exposure_setting_exp3": ExposureSetting.EXP3.value,
    "backbone_variants": ["3dgs", "scaffold_gs"],
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_exp3_ours_dagger": TABLE1_EXP3_OURS_DAGGER,
        "table1_exp3_hdr_gs": TABLE1_EXP3_HDR_GS,
        "table2_exp3_syn": TABLE2_EXP3,
        "table3_efficiency": TABLE3_EFFICIENCY,
        "table4_ablation": TABLE4_ABLATION,
        "hdr_psnr_gain_over_hdr_gs_db": PSNR_GAIN_OVER_HDR_GS,
        "datasets": DATASETS,
        "exposure_times": list(EXPOSURE_TIMES),
        "training_defaults": TRAINING_DEFAULTS,
    }
