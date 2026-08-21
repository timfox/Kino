"""AdaMaG smoke evaluation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.adamag.config import AdamagConfig
from ltx_trainer.adamag.guidance import adamag_velocity, cfg_velocity, omega_schedule


def evaluation_smoke() -> dict[str, Any]:
    cfg = AdamagConfig()
    v_u = [0.1, 0.2]
    v_c = [0.3, 0.4]
    omega_ref = cfg.sd3_high_guidance_omega
    omega_min = cfg.omega_min_default
    x = [0.1, 0.5]
    omega_early = omega_schedule(0.05, omega_ref, omega_min=omega_min, gamma=cfg.gamma_default)
    omega_late = omega_schedule(0.95, omega_ref, omega_min=omega_min, gamma=cfg.gamma_default)
    cfg_norm = sum(v * v for v in cfg_velocity(v_u, v_c, omega_early)) ** 0.5
    ada_v = adamag_velocity(
        v_u, v_c, x, t=0.5, omega_ref=omega_ref, beta=cfg.beta_default, gamma=cfg.gamma_default, omega_min=omega_min
    )
    adamag_norm = sum(v * v for v in ada_v) ** 0.5
    return {
        "dim": len(v_u),
        "omega_early": round(omega_early, 4),
        "omega_late": round(omega_late, 4),
        "cfg_norm": round(cfg_norm, 4),
        "adamag_norm": round(adamag_norm, 4),
        "late_attenuates": omega_late <= omega_early,
    }
