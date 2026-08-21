"""AdaMaG guidance schedule for inference (arXiv:2605.20079).

Use when ``GOPEX_INFER_ADAMAG=1`` to modulate effective CFG strength over denoising steps
without extra NFE (probability-conserving guidance vs linear CFG extrapolation).
"""

from __future__ import annotations

from ltx_trainer.adamag.config import AdamagConfig
from ltx_trainer.adamag.guidance import omega_schedule


def effective_guidance_scale(
    base_scale: float,
    *,
    step_index: int,
    total_steps: int,
    cfg: AdamagConfig | None = None,
) -> float:
    """Scale CFG ω by AdaMaG ``ω(t)`` relative to reference high-guidance ω."""
    c = cfg or AdamagConfig()
    if total_steps <= 1:
        t = 0.5
    else:
        t = 1.0 - float(step_index) / float(total_steps - 1)
    omega_t = omega_schedule(
        t,
        c.sd3_high_guidance_omega,
        omega_min=c.omega_min_default,
        gamma=c.gamma_default,
    )
    ref = c.sd3_high_guidance_omega
    return float(base_scale) * (omega_t / ref)
