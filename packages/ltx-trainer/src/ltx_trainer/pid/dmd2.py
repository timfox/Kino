"""DMD2 four-step distillation schedule (Sec. 3.4, 4.2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pid.config import PiDConfig


def student_sigma_schedule(cfg: PiDConfig | None = None) -> tuple[float, ...]:
    cfg = cfg or PiDConfig()
    return cfg.dmd2_sigmas


def distillation_card() -> dict[str, Any]:
    """Training hyper-parameters from Sec. 4.2 distillation."""
    cfg = PiDConfig()
    return {
        "method": "DMD2",
        "student_steps": cfg.student_steps,
        "sigma_schedule": list(cfg.dmd2_sigmas),
        "dmd_weight": 1.0,
        "dsm_weight": 1.0,
        "gan_weight": 0.05,
        "r1_weight": 200.0,
        "lr": cfg.distill_lr,
        "weight_decay": 1e-3,
        "iterations": 3000,
        "batch_size": 16,
        "context_parallel": 8,
        "note": "CFG distilled into student; retains noisy-latent conditioning",
    }
