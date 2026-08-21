"""LDM early termination M/N for PiD(M/N) decoding (Sec. 3.4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pid.config import PiDConfig


def latent_noise_from_termination(m: int, n: int, *, sigma_max: float = 0.8) -> float:
    """Map LDM step m of n to latent noise σ for PiD conditioning."""
    if n <= 0:
        return 0.0
    frac = max(0.0, min(1.0, 1.0 - m / n))
    return frac * sigma_max


def pid_label(m: int, n: int) -> str:
    return f"PiD({m}/{n})"


def early_exit_plan(
    preset: str = "flux1_dev",
    *,
    cfg: PiDConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PiDConfig()
    m, n = cfg.early_exit_presets.get(preset, (24, 28))
    return {
        "preset": preset,
        "m": m,
        "n": n,
        "label": pid_label(m, n),
        "latent_sigma": latent_noise_from_termination(m, n, sigma_max=cfg.sigma_max),
        "note": "Last 3–5 LDM steps often optimal (Fig. 8)",
    }
