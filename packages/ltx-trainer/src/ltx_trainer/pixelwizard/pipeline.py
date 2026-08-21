"""PixelWizard framework card, paper tables, and smoke demos (arXiv:2605.25801)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pixelwizard.anchor import degrade_anchor
from ltx_trainer.pixelwizard.config import PixelWizardConfig
from ltx_trainer.pixelwizard.losses import flow_matching_loss, shortcut_training_loss
from ltx_trainer.pixelwizard.model import PixelWizardVelocityHead
from ltx_trainer.pixelwizard.shortcut import (
    calibration_weight,
    candidate_step_sizes,
    select_shortcut_step,
    shortcut_step,
)


def framework_card(cfg: PixelWizardConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PixelWizardConfig()
    return {
        "name": "PixelWizard",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "base_model": cfg.base_model,
        "stages": [
            "Spatial-Temporal Anchor Modeling (448×256)",
            "Anchor-Guided High-Resolution Synthesis (2K/4K)",
        ],
        "acceleration": "Noise-Span Aligned Shortcut Training (distillation-free)",
        "hr_inference_steps": cfg.hr_inference_steps,
        "anchor_tune_samples": cfg.anchor_tune_samples,
        "hr_dataset": f"UltraVideo-4K (~{cfg.hr_train_videos // 1000}k videos)",
    }


def table_vbench_hr() -> list[dict[str, Any]]:
    """Table 1 — VBench + latency on 100 prompts."""
    rows = [
        ("Wan2.2-TI2V-5b", "121×2560×1440", 75.29, 24, 1635, 3.66e-6, 13.51),
        ("Wan2.2-TI2V-5b", "121×3840×2144", 68.59, 24, 7150, 7.18e-6, 59.09),
        ("FlashVideo", "49×1920×1072", 78.16, 8, 124, 1.23e-6, 2.53),
        ("UltraWan-4k", "81×3840×2160", 71.44, 16, 24125, 3.59e-5, 298.17),
        ("CineScale-Pro", "81×2880×1632", 79.25, 16, 15235, 4.00e-5, 188.09),
        ("PixelWizard-2k", "121×2560×1440", 79.34, 24, 146, 3.27e-7, 1.21),
        ("PixelWizard-4k", "121×3840×2144", 79.62, 24, 590, 5.92e-7, 4.88),
    ]
    return [
        {
            "model": m,
            "resolution": r,
            "vbench_avg": avg,
            "fps": fps,
            "latency_sec": lat,
            "latency_per_pixel": lpp,
            "latency_per_frame": lpf,
        }
        for m, r, avg, fps, lat, lpp, lpf in rows
    ]


def table_detail_quality() -> dict[str, dict[str, float]]:
    """Table 2 — mHD-MSE, mHD-LPIPS, DOVER."""
    return {
        "FlashVideo": {"mhd_mse": 0.0098, "mhd_lpips": 0.4756, "tech": 12.47, "aesth": 98.11},
        "UltraWan-4k": {"mhd_mse": 0.0012, "mhd_lpips": 0.3319, "tech": 10.59, "aesth": 98.88},
        "CineScale-Pro": {"mhd_mse": 0.0050, "mhd_lpips": 0.3483, "tech": 12.76, "aesth": 98.15},
        "PixelWizard-2k": {"mhd_mse": 0.0147, "mhd_lpips": 0.5206, "tech": 14.31, "aesth": 99.58},
        "PixelWizard-4k": {"mhd_mse": 0.0102, "mhd_lpips": 0.5120, "tech": 14.14, "aesth": 99.66},
    }


def table_video_sr() -> dict[str, dict[str, float]]:
    """Table 3 — comparison with SR pipelines."""
    return {
        "STAR@2k": {"musiq": 37.80, "niqe": 5.55, "mhd_mse": 0.0040, "tech": 11.47},
        "DOVE@2k": {"musiq": 50.54, "niqe": 4.92, "mhd_mse": 0.0050, "tech": 12.31},
        "FlashVSR@2k": {"musiq": 42.84, "niqe": 5.53, "mhd_mse": 0.0076, "tech": 13.52},
        "PixelWizard-2k": {"musiq": 57.67, "niqe": 3.94, "mhd_mse": 0.0147, "tech": 14.31},
        "FlashVSR@4k": {"musiq": 43.93, "niqe": 4.32, "mhd_mse": 0.0053, "tech": 12.80},
        "PixelWizard-4k": {"musiq": 48.17, "niqe": 4.19, "mhd_mse": 0.0102, "tech": 14.14},
    }


def table_ablation_aghs() -> dict[str, dict[str, float]]:
    """Table 4 — Anchor-Guided High-Resolution Synthesis."""
    return {
        "Anchor": {"tech": 8.01, "aesth": 97.91},
        "Anchor w/o Tuning": {"tech": 3.30, "aesth": 79.11},
        "+AGHS w/o AGI": {"tech": 12.51, "aesth": 98.99},
        "+AGHS (α=0)": {"tech": 13.47, "aesth": 99.36},
        "+AGHS": {"tech": 14.31, "aesth": 99.58},
    }


def table_ablation_inference_steps() -> dict[int, dict[str, float]]:
    """Table 5 — HR inference steps vs quality/latency."""
    return {
        2: {"tech": 13.73, "aesth": 99.03, "latency_sec": 99},
        3: {"tech": 14.31, "aesth": 99.56, "latency_sec": 132},
        4: {"tech": 14.39, "aesth": 99.58, "latency_sec": 165},
        5: {"tech": 14.38, "aesth": 99.59, "latency_sec": 264},
    }


def table_vbench_long() -> dict[str, dict[str, float]]:
    """Table 8.1 — VBench-Long (selected)."""
    return {
        "Turbo2K": {"total": 82.78, "quality": 84.91, "semantic": 74.24},
        "FlashVideo": {"total": 82.80, "quality": 82.99, "semantic": 82.03},
        "PixelWizard-2k": {"total": 83.62, "quality": 84.56, "semantic": 79.86},
    }


def table_memory() -> dict[str, dict[str, float | str]]:
    """Table 8.2 — 4K peak memory."""
    return {
        "FlashVideo (4K)": {"params": "7B", "peak_mem": "OOM"},
        "UltraWan-4K": {"params": "1.3B", "peak_mem": "OOM"},
        "PixelWizard-4K": {"params": "10B", "peak_mem_gb": 101.8},
    }


def table_ablation_shortcut() -> dict[str, dict[str, float]]:
    """Table 9.3 — Noise-Span Aligned Shortcut (4-step)."""
    return {
        "Baseline": {"musiq": 51.54, "niqe": 4.04, "tech": 12.98, "aesth": 99.01},
        "Shortcut Original Sampling": {"musiq": 53.08, "niqe": 4.02, "tech": 13.31, "aesth": 99.24},
        "Exponential Index-based Sampling": {"musiq": 56.43, "niqe": 3.99, "tech": 13.88, "aesth": 99.52},
        "Full PixelWizard-2k": {"musiq": 57.67, "niqe": 3.94, "tech": 14.31, "aesth": 99.58},
    }


def training_step_demo(cfg: PixelWizardConfig | None = None) -> dict[str, float]:
    """Smoke: anchor degrade + velocity head + flow / shortcut losses."""
    cfg = cfg or PixelWizardConfig()
    torch.manual_seed(5)
    b, c, h, w = 2, 16, 32, 32
    x1 = torch.rand(b, c, h, w)
    x0 = torch.randn(b, c, h, w)
    anchor = degrade_anchor(x1)
    t = torch.tensor([0.6, 0.65])
    dt_val = select_shortcut_step(600, k=cfg.shortcut_candidates_k, beta=cfg.exponential_beta)
    model = PixelWizardVelocityHead(in_channels=c, hidden=32)
    v = model(x0, anchor, t, torch.tensor([float(dt_val), float(dt_val)]))
    lf = flow_matching_loss(v, x0, x1, t)
    ls, parts = shortcut_training_loss(lambda xt, tt, d: model(xt, anchor, tt, d), x0, t, dt_val, cfg)
    x_step = shortcut_step(x0, v, torch.tensor([0.05, 0.05]))
    return {
        "flow_loss": float(lf.detach()),
        "shortcut_loss": float(ls.detach()),
        "shortcut_step": float(dt_val),
        "lambda_anc": parts.get("lambda_anc", 0.0),
        "candidates_at_600": float(len(candidate_step_sizes(600, cfg.shortcut_candidates_k))),
        "step_output_shape": float(x_step.shape[-1]),
    }


def evaluation_demo(cfg: PixelWizardConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering and efficiency claims."""
    cfg = cfg or PixelWizardConfig()
    step = training_step_demo(cfg)
    t1 = {r["model"]: r for r in table_vbench_hr()}
    pw2k = t1["PixelWizard-2k"]
    wan4k = t1["Wan2.2-TI2V-5b"]
    ultra4k = next(r for r in table_vbench_hr() if r["model"] == "UltraWan-4k")
    detail = table_detail_quality()
    aghs = table_ablation_aghs()
    shortcut = table_ablation_shortcut()
    steps = table_ablation_inference_steps()

    return {
        **step,
        "pw2k_faster_than_wan4k": pw2k["latency_sec"] < wan4k["latency_sec"] / 10,
        "pw4k_faster_than_ultrawan4k": t1["PixelWizard-4k"]["latency_sec"] < ultra4k["latency_sec"] / 10,
        "pw2k_best_vbench_avg": pw2k["vbench_avg"] >= max(
            r["vbench_avg"] for r in table_vbench_hr() if "PixelWizard" not in r["model"]
        ),
        "pw2k_best_mhd_lpips": detail["PixelWizard-2k"]["mhd_lpips"]
        > detail["UltraWan-4k"]["mhd_lpips"],
        "aghs_monotonic_tech": aghs["+AGHS"]["tech"] > aghs["+AGHS w/o AGI"]["tech"] > aghs["Anchor"]["tech"],
        "shortcut_full_best_musiq": shortcut["Full PixelWizard-2k"]["musiq"]
        > shortcut["Baseline"]["musiq"],
        "optimal_steps_4": steps[4]["tech"] >= steps[2]["tech"],
        "vbench_long_beats_turbo2k": table_vbench_long()["PixelWizard-2k"]["total"]
        > table_vbench_long()["Turbo2K"]["total"],
        "calibration_weight_600": calibration_weight(600, select_shortcut_step(600)),
    }
