"""PiD evaluation demos and framework card."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pid.benchmarks import benchmarks_bundle
from ltx_trainer.pid.cascade import cascaded_decode
from ltx_trainer.pid.config import PiDConfig
from ltx_trainer.pid.decode import pid_decode_step
from ltx_trainer.pid.dmd2 import distillation_card
from ltx_trainer.pid.early_exit import early_exit_plan
from ltx_trainer.pid.metrics import metric_bundle


def knowledge_card() -> dict[str, Any]:
    cfg = PiDConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "authors": "Yifan Lu, Qi Wu, Jay Zhangjie Wu, Zian Wang, Huan Ling, Sanja Fidler, Xuanchi Ren",
        "org": "NVIDIA",
        "website": cfg.website,
        "method": (
            "Latent-conditioned pixel diffusion decoder on PixelDiT prior; "
            "sigma-aware adapter; DMD2 4-step student; unifies decode+4×/8× SR"
        ),
        "latent_spaces": [
            "FLUX.1/2 VAE",
            "SD3 VAE",
            "Z-Image VAE",
            "DINOv2 (RAE)",
            "SigLIP (Scale-RAE)",
        ],
        "ltx_hook": cfg.ltx_hook,
    }


def framework_card() -> dict[str, Any]:
    cfg = PiDConfig()
    return {
        "name": cfg.name,
        "config": {
            "base_model": cfg.base_model,
            "student_steps": cfg.student_steps,
            "dmd2_sigmas": list(cfg.dmd2_sigmas),
            "sigma_max": cfg.sigma_max,
            "default_scale": cfg.default_scale,
        },
        "training": {
            "prior": "Rectified flow on MultiAspect-4K-1M (~2.6M)",
            "adapter": "Joint fine-tune backbone + ControlNet-style adapter",
            "distill": distillation_card(),
        },
        "inference": {
            "early_exit": list(cfg.early_exit_presets.items()),
            "latency_anchor_gb200_2048_ms": cfg.latency_gb200_2048_compile_ms,
        },
        "benchmarks": benchmarks_bundle(),
        "knowledge": knowledge_card(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = PiDConfig()
    latent = torch.randn(1, 4, 64, 64)
    scale = 4

    pid_out = pid_decode_step(latent, scale=scale, steps=cfg.student_steps, cfg=cfg)
    cascade = cascaded_decode(latent, scale=scale)

    exit_plan = early_exit_plan("flux1_dev", cfg=cfg)
    pid_partial = pid_decode_step(
        latent,
        scale=scale,
        sigma_latent=exit_plan["latent_sigma"],
        steps=cfg.student_steps,
        cfg=cfg,
    )

    return {
        "shapes": {
            "latent": list(latent.shape),
            "pid_2048": list(pid_out.shape),
            "cascade_low": list(cascade["low_res"].shape),
            "cascade_high": list(cascade["high_res"].shape),
            "pid_early_exit": list(pid_partial.shape),
        },
        "early_exit": exit_plan,
        "metrics_pid": metric_bundle(pid_out),
        "metrics_cascade": metric_bundle(cascade["high_res"]),
        "table1_anchor": benchmarks_bundle()["table1"]["flux1_vae_flux1_dev_pid_24_28"],
        "distillation": distillation_card(),
    }
