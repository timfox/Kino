"""On-the-fly stochastic pulse signal generation (§2.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.spectcount.config import SpectCountConfig


def trapezoidal_window(
    t: np.ndarray,
    *,
    duration: float,
    attack: float,
    release: float,
) -> np.ndarray:
    w = np.zeros_like(t)
    td = duration
    ta, tr = attack, release
    rising = (t >= 0) & (t < ta)
    flat = (t >= ta) & (t < td - tr)
    falling = (t >= td - tr) & (t < td)
    w[rising] = t[rising] / max(ta, 1e-8)
    w[flat] = 1.0
    w[falling] = (td - t[falling]) / max(tr, 1e-8)
    return w


def synthesize_pulse(
    *,
    freq_hz: float,
    duration_s: float,
    amplitude: float,
    sample_rate: int,
    attack_s: float,
    release_s: float,
    phase: float,
) -> np.ndarray:
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    w = trapezoidal_window(t, duration=duration_s, attack=attack_s, release=release_s)
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase) * w


def generate_signal(
    *,
    seed: int = 0,
    cfg: SpectCountConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Superpose N pulses + AWGN; return waveform and count label."""
    cfg = cfg or SpectCountConfig()
    rng = np.random.default_rng(seed)
    sr = cfg.sample_rate_hz
    total_samples = int(cfg.t_total_s * sr)
    x = np.zeros(total_samples, dtype=np.float64)

    n_pulses = int(rng.integers(1, cfg.n_max + 1))
    placed: list[tuple[int, int]] = []
    gap = int(cfg.t_gap_ms / 1000 * sr)

    mel_centers = np.geomspace(200, 7000, cfg.cmel)

    for _ in range(n_pulses):
        for _attempt in range(100):
            fi = float(rng.choice(mel_centers))
            td = rng.uniform(cfg.t_min_ms, cfg.t_max_ms) / 1000.0
            ai = float(np.exp(rng.uniform(np.log(cfg.amp_min), np.log(cfg.amp_max))))
            phi = float(rng.uniform(0, 2 * np.pi))
            pulse = synthesize_pulse(
                freq_hz=fi,
                duration_s=td,
                amplitude=ai,
                sample_rate=sr,
                attack_s=cfg.attack_ms / 1000,
                release_s=cfg.release_ms / 1000,
                phase=phi,
            )
            plen = len(pulse)
            if plen >= total_samples:
                continue
            tau = int(rng.integers(0, total_samples - plen))
            ok = all(abs(tau - p0) >= gap and abs(tau + plen - p1) >= gap for p0, p1 in placed)
            if tau + plen <= total_samples and ok:
                x[tau : tau + plen] += pulse
                placed.append((tau, tau + plen))
                break

    sigma = float(np.exp(rng.uniform(np.log(cfg.noise_min), np.log(cfg.noise_max))))
    x += rng.normal(0, sigma, size=total_samples)
    return np.clip(x, -1.0, 1.0), n_pulses


def signal_demo(*, seed: int = 0, cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    wave, count = generate_signal(seed=seed, cfg=cfg)
    return {
        "sample_rate_hz": cfg.sample_rate_hz,
        "duration_s": cfg.t_total_s,
        "pulse_count": count,
        "waveform_samples": len(wave),
        "n_max": cfg.n_max,
        "cmel_channels": cfg.cmel,
    }
