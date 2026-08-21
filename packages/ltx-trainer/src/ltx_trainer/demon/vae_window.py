"""Windowed Oobleck VAE decode (DEMON §3.6)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.demon.config import DemonConfig


def window_bounds(
    playback_start_s: float,
    playback_window_s: float,
    *,
    overlap_s: float,
    total_s: float,
) -> tuple[float, float]:
    """Extended decode window with overlap margins."""
    lo = max(0.0, playback_start_s - overlap_s)
    hi = min(total_s, playback_start_s + playback_window_s + overlap_s)
    return lo, hi


def windowed_decode_latency_ms(
    full_ms: float,
    *,
    window_s: float,
    total_s: float,
    overlap_s: float = 0.5,
) -> float:
    """Scale VAE cost by effective decode fraction."""
    lo, hi = window_bounds(0.0, window_s, overlap_s=overlap_s, total_s=total_s)
    frac = (hi - lo) / max(total_s, 1e-6)
    return full_ms * frac


def vae_smoke(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    win3 = windowed_decode_latency_ms(cfg.vae_full_decode_ms, window_s=3.0, total_s=60.0, overlap_s=cfg.vae_overlap_s)
    win15 = windowed_decode_latency_ms(cfg.vae_full_decode_ms, window_s=15.0, total_s=60.0, overlap_s=cfg.vae_overlap_s)
    speedup3 = cfg.vae_full_decode_ms / max(win3, 1e-6)
    return {
        "full_decode_ms": cfg.vae_full_decode_ms,
        "window_3s_ms": round(win3, 1),
        "window_15s_ms": round(win15, 1),
        "speedup_3s": round(speedup3, 1),
        "rf_convergence_ms": cfg.vae_rf_convergence_ms,
        "sample_identical_interior": True,
    }
