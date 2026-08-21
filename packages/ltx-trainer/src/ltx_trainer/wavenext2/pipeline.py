"""WaveNeXt 2 framework card and paper tables (arXiv:2605.25506)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.wavenext2.config import Wavenext2Config
from ltx_trainer.wavenext2.diffusion import forward_diffusion_sample, mse_denoising_loss
from ltx_trainer.wavenext2.gan import fixed_point_iteration, model_size_millions
from ltx_trainer.wavenext2.layout import LIMITATIONS
from ltx_trainer.wavenext2.mock import toy_diff_synthesize, toy_gan_synthesize


def framework_card(cfg: Wavenext2Config | None = None) -> dict[str, Any]:
    cfg = cfg or Wavenext2Config()
    return {
        "name": "WaveNeXt 2",
        "paper": cfg.paper_arxiv,
        "demo": cfg.demo_page,
        "idea": (
            "Unified ConvNeXt-based generator with residual denoising sub-modeling "
            "for both GAN (fixed-point, WaveFit-style) and diffusion (BDDM 4-step, "
            "noise-level-limited sub-models) vocoders on LibriTTS-R."
        ),
        "variants": ["GAN-WaveNeXt 2", "Diff-WaveNeXt 2"],
        "convnext_blocks": cfg.convnext_blocks,
        "diffusion_submodels": cfg.num_diff_submodels,
        "bddm_schedule": list(cfg.bddm_noise_schedule),
        "training": {
            "GAN-WaveNeXt 2": "~410 h (1× A100)",
            "Diff-WaveNeXt 2": "~32 h (1× A100)",
        },
        "defaults": cfg.__dict__,
    }


def table_i_objective() -> list[dict[str, Any]]:
    """Table 1 — objective metrics + RTF (bold rows = proposed)."""
    return [
        {"model": "Ground Truth", "rtf_gpu": None, "rtf_cpu": None, "nisqa": 4.08, "utmos": 4.11, "mcd": None, "log_f0_rmse": None, "size_m": None},
        {"model": "WaveNeXt (1 iter)", "rtf_gpu": 0.0022, "rtf_cpu": 0.06, "nisqa": 3.16, "utmos": 3.20, "mcd": 0.92, "log_f0_rmse": 0.31, "size_m": 14.98},
        {"model": "WaveFit (2 iter)", "rtf_gpu": 0.0111, "rtf_cpu": 2.15, "nisqa": 3.80, "utmos": 3.89, "mcd": 1.03, "log_f0_rmse": 0.32, "size_m": 15.51},
        {"model": "GAN-WaveNeXt 2 (2 iter)", "proposed": True, "rtf_gpu": 0.0033, "rtf_cpu": 0.10, "nisqa": 3.77, "utmos": 3.88, "mcd": 0.97, "log_f0_rmse": 0.31, "size_m": 29.97},
        {"model": "WaveFit (3 iter)", "rtf_gpu": 0.0151, "rtf_cpu": 3.22, "nisqa": 3.91, "utmos": 3.98, "mcd": 1.01, "log_f0_rmse": 0.32, "size_m": 15.51},
        {"model": "GAN-WaveNeXt 2 (3 iter)", "proposed": True, "rtf_gpu": 0.0054, "rtf_cpu": 0.15, "nisqa": 3.92, "utmos": 3.91, "mcd": 0.96, "log_f0_rmse": 0.30, "size_m": 44.96},
        {"model": "WaveFit (4 iter)", "rtf_gpu": 0.0213, "rtf_cpu": 4.28, "nisqa": 3.97, "utmos": 3.99, "mcd": 1.01, "log_f0_rmse": 0.32, "size_m": 15.51},
        {"model": "GAN-WaveNeXt 2 (4 iter)", "proposed": True, "rtf_gpu": 0.0066, "rtf_cpu": 0.20, "nisqa": 4.01, "utmos": 4.04, "mcd": 0.95, "log_f0_rmse": 0.30, "size_m": 59.94},
        {"model": "WaveFit (5 iter)", "rtf_gpu": 0.0226, "rtf_cpu": 5.36, "nisqa": 4.02, "utmos": 4.04, "mcd": 0.90, "log_f0_rmse": 0.31, "size_m": 15.51},
        {"model": "GAN-WaveNeXt 2 (5 iter)", "proposed": True, "rtf_gpu": 0.0090, "rtf_cpu": 0.24, "nisqa": 4.01, "utmos": 4.04, "mcd": 0.95, "log_f0_rmse": 0.30, "size_m": 74.93},
        {"model": "HiFi-GAN V1", "rtf_gpu": 0.0110, "rtf_cpu": 0.80, "nisqa": 3.99, "utmos": 4.05, "mcd": 2.34, "log_f0_rmse": 0.16, "size_m": 13.9},
        {"model": "FastDiff w/o sub-model", "rtf_gpu": 0.0625, "rtf_cpu": 0.80, "nisqa": 3.43, "utmos": 3.50, "mcd": 4.76, "log_f0_rmse": 0.16, "size_m": 15.63},
        {"model": "Diff-WaveNeXt 2 w/o sub-model", "proposed": True, "rtf_gpu": 0.0335, "rtf_cpu": 0.16, "nisqa": 3.45, "utmos": 3.55, "mcd": 7.34, "log_f0_rmse": 0.16, "size_m": 14.42},
        {"model": "FastDiff w/ sub-model", "rtf_gpu": 0.0282, "rtf_cpu": 0.80, "nisqa": 3.67, "utmos": 3.78, "mcd": 4.32, "log_f0_rmse": 0.24, "size_m": 62.52},
        {"model": "Diff-WaveNeXt 2", "proposed": True, "rtf_gpu": 0.0164, "rtf_cpu": 0.16, "nisqa": 3.81, "utmos": 3.87, "mcd": 4.16, "log_f0_rmse": 0.12, "size_m": 57.68},
    ]


def table_ii_training_hours() -> list[dict[str, Any]]:
    """Table 2 — single-GPU training time."""
    return [
        {"model": "GAN-WaveNeXt 2", "hours": 410},
        {"model": "HiFi-GAN", "hours": 270},
        {"model": "WaveFit", "hours": 410},
        {"model": "Diff-WaveNeXt 2", "hours": 32},
        {"model": "FastDiff", "hours": 96},
    ]


def figure_iv_mos() -> list[dict[str, Any]]:
    """Fig. 4 — MOS (5-point scale, 95% CI, 20 listeners)."""
    return [
        {"model": "Ground Truth", "mos": 4.5},
        {"model": "HiFi-GAN V1", "mos": 4.0},
        {"model": "WaveFit (5 iter)", "mos": 4.1},
        {"model": "GAN-WaveNeXt 2 (4 iter)", "mos": 4.0},
        {"model": "FastDiff w/ sub-model", "mos": 3.6},
        {"model": "Diff-WaveNeXt 2", "mos": 3.8},
        {"model": "WaveNeXt (1 iter)", "mos": 3.2},
    ]


def headline_results() -> dict[str, Any]:
    gan4 = next(r for r in table_i_objective() if r["model"] == "GAN-WaveNeXt 2 (4 iter)")
    wavefit5 = next(r for r in table_i_objective() if r["model"] == "WaveFit (5 iter)")
    diff = next(r for r in table_i_objective() if r["model"] == "Diff-WaveNeXt 2")
    fastdiff = next(r for r in table_i_objective() if r["model"] == "FastDiff w/ sub-model")
    return {
        "gan_cpu_rtf_vs_wavefit5": {
            "gan_wavenext2": gan4["rtf_cpu"],
            "wavefit5": wavefit5["rtf_cpu"],
            "cpu_reduction_pct": round(100 * (1 - gan4["rtf_cpu"] / wavefit5["rtf_cpu"]), 0),
        },
        "diff_cpu_rtf_vs_fastdiff": {
            "diff_wavenext2": diff["rtf_cpu"],
            "fastdiff_sub": fastdiff["rtf_cpu"],
            "cpu_reduction_pct": round(100 * (1 - diff["rtf_cpu"] / fastdiff["rtf_cpu"]), 0),
        },
        "diff_training_hours": 32,
        "fastdiff_training_hours": 96,
        "recommended": {
            "resource_constrained": "Diff-WaveNeXt 2",
            "highest_quality": "GAN-WaveNeXt 2 (4–5 iter)",
        },
    }


def evaluation_demo(cfg: Wavenext2Config | None = None) -> dict[str, Any]:
    cfg = cfg or Wavenext2Config()
    rng = np.random.default_rng(11)
    mel = rng.standard_normal(cfg.mel_dim)
    x0 = rng.standard_normal(512) * 0.05
    x_t = forward_diffusion_sample(x0, cfg.bddm_noise_schedule[2], rng)
    gan_out = toy_gan_synthesize(mel, iterations=4, seed=1)
    diff_out = toy_diff_synthesize(mel, cfg.bddm_noise_schedule, seed=2)
    y4 = fixed_point_iteration(mel, 4, rng)
    return {
        "convnext_blocks": cfg.convnext_blocks,
        "bddm_schedule": list(cfg.bddm_noise_schedule),
        "forward_diffusion_sample_len": len(x_t),
        "mse_toy_loss": mse_denoising_loss(x0, x0 - 0.1 * rng.standard_normal(len(x0))),
        "gan_4iter": gan_out,
        "diff_4submodel": diff_out,
        "gan_waveform_rms": float(np.sqrt(np.mean(y4**2))),
        "gan_size_4iter_m": model_size_millions(4),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_objective": table_i_objective(),
        "table_ii_training_hours": table_ii_training_hours(),
        "figure_iv_mos": figure_iv_mos(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
