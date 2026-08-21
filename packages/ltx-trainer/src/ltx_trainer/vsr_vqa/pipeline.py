"""QoMEX VSR-VQA framework card, Table I, and correlation smoke demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.vsr_vqa.config import VSRVQAConfig
from ltx_trainer.vsr_vqa.correlation import (
    fisher_z_mean,
    pearson_correlation,
    rmse,
    spearman_correlation,
)


def framework_card(cfg: VSRVQAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VSRVQAConfig()
    return {
        "name": "AVT-VQDB-UHD-1-VSR",
        "paper": cfg.paper_arxiv,
        "conference": cfg.conference,
        "title": "How Accurate are Video Quality Models for Diffusion-Based Video Super-Resolution?",
        "pvs": cfg.num_pvs,
        "participants": f"{cfg.num_participants_valid}/{cfg.num_participants}",
        "upscale_methods": list(cfg.upscale_methods),
        "codecs": list(cfg.codecs),
        "focus": "within-sequence correlation (Fisher z average over 6 sources)",
        "dataset": cfg.dataset_repo,
        "conclusion": "CNN FR models best; none replace subjective testing",
    }


def table_within_sequence_correlation() -> dict[str, dict[str, float]]:
    """Table I — mean within-sequence PLCC / SRCC / RMSE (×100 SRCC in fig uses same values)."""
    return {
        "PSNR": {"plcc": 0.497, "srcc": 0.507, "rmse": 0.845},
        "PSNR-HVS": {"plcc": 0.510, "srcc": 0.539, "rmse": 0.841},
        "SSIM": {"plcc": 0.785, "srcc": 0.762, "rmse": 0.603},
        "MS-SSIM": {"plcc": 0.725, "srcc": 0.722, "rmse": 0.672},
        "SSIMULACRA2": {"plcc": 0.660, "srcc": 0.750, "rmse": 0.740},
        "Butteraugli": {"plcc": 0.510, "srcc": 0.481, "rmse": 0.843},
        "VMAF": {"plcc": 0.653, "srcc": 0.637, "rmse": 0.731},
        "VMAF_NEG": {"plcc": 0.715, "srcc": 0.713, "rmse": 0.680},
        "CVVDP": {"plcc": 0.766, "srcc": 0.729, "rmse": 0.626},
        "PIEAPP": {"plcc": 0.632, "srcc": 0.686, "rmse": 0.759},
        "LPIPS_Alex": {"plcc": 0.851, "srcc": 0.881, "rmse": 0.535},
        "LPIPS_VGG": {"plcc": 0.810, "srcc": 0.880, "rmse": 0.583},
        "DISTS": {"plcc": 0.838, "srcc": 0.850, "rmse": 0.551},
        "CVQA-FR": {"plcc": 0.845, "srcc": 0.847, "rmse": 0.525},
        "CVQA-FR-MS": {"plcc": 0.856, "srcc": 0.853, "rmse": 0.510},
        "BRISQUE": {"plcc": 0.466, "srcc": 0.501, "rmse": 0.868},
        "NIQE": {"plcc": 0.425, "srcc": 0.427, "rmse": 0.867},
        "MUSIQ": {"plcc": 0.500, "srcc": 0.476, "rmse": 0.813},
        "CLIP-IQA+": {"plcc": 0.428, "srcc": 0.441, "rmse": 0.853},
        "MDTVSFA": {"plcc": 0.451, "srcc": 0.529, "rmse": 0.826},
        "UVQ": {"plcc": 0.327, "srcc": 0.281, "rmse": 0.886},
        "UVQ-1.5": {"plcc": 0.507, "srcc": 0.602, "rmse": 0.828},
        "CVQA-NR": {"plcc": 0.603, "srcc": 0.665, "rmse": 0.767},
        "CVQA-NR-MS": {"plcc": 0.630, "srcc": 0.655, "rmse": 0.734},
        "FAST-VQA": {"plcc": 0.546, "srcc": 0.579, "rmse": 0.817},
        "FasterVQA": {"plcc": 0.602, "srcc": 0.683, "rmse": 0.775},
        "Dover": {"plcc": 0.520, "srcc": 0.549, "rmse": 0.837},
        "MaxVQA": {"plcc": 0.430, "srcc": 0.505, "rmse": 0.868},
        "Q-Align": {"plcc": 0.116, "srcc": 0.121, "rmse": 0.949},
        "Cover": {"plcc": 0.546, "srcc": 0.592, "rmse": 0.816},
    }


def table_overall_correlation() -> dict[str, dict[str, float]]:
    """Table I — overall (across all PVS) excerpt for top/bottom metrics."""
    return {
        "LPIPS_Alex": {"plcc": 0.609, "srcc": 0.636, "rmse": 0.808},
        "CVQA-FR-MS": {"plcc": 0.733, "srcc": 0.721, "rmse": 0.693},
        "DISTS": {"plcc": 0.712, "srcc": 0.712, "rmse": 0.715},
        "FasterVQA": {"plcc": 0.556, "srcc": 0.538, "rmse": 0.847},
        "Q-Align": {"plcc": 0.134, "srcc": 0.108, "rmse": 1.010},
    }


def subjective_method_ranking() -> dict[str, str]:
    """Sec. III qualitative ranking."""
    return {
        "best_overall": "SeedVR2 / DOVE / Starlight Mini (comparable)",
        "worst_overall": "SCST",
        "best_uncompressed_360p": "SeedVR2 near source MOS",
        "lanczos_role": "baseline conventional upscaling",
    }


def top_fr_nr_within_sequence() -> dict[str, list[str]]:
    """Top-3 FR and best NR from Table I caption."""
    fr = sorted(
        table_within_sequence_correlation().items(),
        key=lambda kv: kv[1]["srcc"],
        reverse=True,
    )
    nr_only = {
        k: v
        for k, v in table_within_sequence_correlation().items()
        if k
        in {
            "BRISQUE",
            "NIQE",
            "MUSIQ",
            "CLIP-IQA+",
            "MDTVSFA",
            "UVQ",
            "UVQ-1.5",
            "CVQA-NR",
            "CVQA-NR-MS",
            "FAST-VQA",
            "FasterVQA",
            "Dover",
            "MaxVQA",
            "Q-Align",
            "Cover",
        }
    }
    nr_best = max(nr_only.items(), key=lambda kv: kv[1]["srcc"])
    return {
        "fr_top3_srcc": [k for k, _ in fr[:3]],
        "nr_best_srcc": nr_best[0],
    }


def training_step_demo(cfg: VSRVQAConfig | None = None) -> dict[str, float]:
    """Smoke: correlate synthetic MOS with LPIPS-like objective scores."""
    cfg = cfg or VSRVQAConfig()
    torch.manual_seed(3)
    mos = torch.tensor([2.1, 2.8, 3.4, 3.9, 4.2, 4.5])  # higher = better perceived quality
    # FR metrics often anti-correlate with MOS for VSR (lower distance = higher quality)
    lpips_like = 5.0 - mos + torch.randn(6) * 0.05
    plcc = pearson_correlation(lpips_like, mos)
    srcc = spearman_correlation(lpips_like, mos)
    err = rmse(lpips_like, mos)

    per_source = [0.85, 0.88, 0.84, 0.90, 0.86, 0.87]
    fisher_mean = fisher_z_mean(per_source)

    tab = table_within_sequence_correlation()
    return {
        "demo_plcc": plcc,
        "demo_srcc": srcc,
        "demo_rmse": err,
        "fisher_z_mean_srcc": fisher_mean,
        "table_lpips_alex_srcc": tab["LPIPS_Alex"]["srcc"],
        "table_faster_vqa_srcc": tab["FasterVQA"]["srcc"],
        "num_pvs": float(cfg.num_pvs),
    }


def evaluation_demo() -> dict[str, Any]:
    cfg = VSRVQAConfig()
    within = table_within_sequence_correlation()
    tops = top_fr_nr_within_sequence()
    step = training_step_demo(cfg)

    lpips = within["LPIPS_Alex"]
    cvqa = within["CVQA-FR-MS"]
    vmaf = within["VMAF"]
    faster = within["FasterVQA"]
    qalign = within["Q-Align"]

    return {
        **step,
        "lpips_alex_srcc_within": lpips["srcc"],
        "cvqa_fr_ms_srcc_within": cvqa["srcc"],
        "faster_vqa_srcc_within": faster["srcc"],
        "qalign_srcc_within": qalign["srcc"],
        "cnn_fr_beats_ssim": lpips["srcc"] > within["SSIM"]["srcc"],
        "nr_below_fr": faster["srcc"] < cvqa["srcc"],
        "vmaf_neg_beats_vmaf": within["VMAF_NEG"]["srcc"] > vmaf["srcc"],
        "fr_top3": tops["fr_top3_srcc"],
        "nr_best": tops["nr_best_srcc"],
        "scst_slowest_spf": cfg.processing_seconds_per_frame["SCST"],
        "sos_a": cfg.sos_a,
    }
