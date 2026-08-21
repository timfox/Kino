"""Reference metrics — LF-Diff (arXiv:2503.07351)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2503.07351"
PAPER_TITLE = "LF-Diff: Linearized Diffusion for HDR Image Reconstruction"

TABLE1_SYNTHETIC: dict[str, float] = {"pu21_psnr": 38.2, "ssim": 0.981, "mu_inf": 5000.0}

TABLE2_REAL: dict[str, dict[str, float]] = {
    "lf_diff": {"pu21_psnr": 36.8, "lpips": 0.042, "hdr_vdp": 7.2},
    "hdrtvnet": {"pu21_psnr": 28.1, "lpips": 0.095, "hdr_vdp": 5.8},
    "kalantari": {"pu21_psnr": 32.4, "lpips": 0.068, "hdr_vdp": 6.5},
}

TABLE3_ABLATION: dict[str, dict[str, float]] = {
    "full": {"tonemap_l1": 0.012, "lpr_l1": 0.008},
    "no_lpr": {"tonemap_l1": 0.019, "lpr_l1": 0.0},
    "no_tonemap_loss": {"tonemap_l1": 0.028, "lpr_l1": 0.011},
}

TRAINING_DEFAULTS = {
    "steps": 50_000,
    "bracket_count": 3,
    "tonemap_mu": 5000.0,
    "denoise_steps": 10,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_synthetic": TABLE1_SYNTHETIC,
        "table2_real": TABLE2_REAL,
        "table3_ablation": TABLE3_ABLATION,
        "training_defaults": TRAINING_DEFAULTS,
    }
