"""Acoustic physical metrics (Sec. 3.2) — numpy stubs for benchmark smoke."""

from __future__ import annotations

import numpy as np


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, sr // 200)
    frame = hop * 4
    n = max(1, (len(y) - frame) // hop + 1)
    env = np.zeros(n)
    for i in range(n):
        seg = y[i * hop : i * hop + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        env[i] = np.sqrt(np.mean(seg**2) + 1e-12)
    return env


def attack_time_ms(y: np.ndarray, sr: int) -> float:
    env = _envelope(y, sr)
    if env.size < 3:
        return float("nan")
    peak = float(env.max())
    i10 = next((i for i, v in enumerate(env) if v >= 0.1 * peak), 0)
    i90 = next((i for i, v in enumerate(env) if v >= 0.9 * peak), env.size - 1)
    hop_s = 128.0 / sr
    return 1000.0 * max(1, i90 - i10) * hop_s


def decay_rate(y: np.ndarray, sr: int) -> float:
    env = _envelope(y, sr)
    if env.size < 4:
        return float("nan")
    peak = float(env.max())
    tail = env[env >= 0.1 * peak]
    if tail.size < 2:
        return float("nan")
    t = np.arange(tail.size) / (sr / 128)
    loge = np.log(tail + 1e-9)
    slope = np.polyfit(t, loge, 1)[0]
    return float(max(0.02, min(50.0, -slope)))


def fundamental_frequency_hz(y: np.ndarray, sr: int) -> float:
    n = min(len(y), sr // 2)
    if n < 256:
        return float("nan")
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 80) & (freqs <= 2000)
    if not np.any(band):
        return float("nan")
    k = int(np.argmax(spec[band]))
    return float(freqs[band][k])


def spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    n = min(len(y), sr // 4)
    if n < 128:
        return float("nan")
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    denom = spec.sum() + 1e-12
    return float(np.sum(freqs * spec) / denom)


def spectral_rolloff_hz(y: np.ndarray, sr: int, roll: float = 0.85) -> float:
    n = min(len(y), sr // 4)
    if n < 128:
        return float("nan")
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    cum = np.cumsum(spec)
    target = roll * cum[-1]
    idx = int(np.searchsorted(cum, target))
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_flux(y: np.ndarray, sr: int) -> float:
    hop = 128
    frame = 512
    n_frames = max(1, (len(y) - frame) // hop + 1)
    prev = None
    fluxes: list[float] = []
    for i in range(n_frames):
        seg = y[i * hop : i * hop + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        mag = np.abs(np.fft.rfft(seg * np.hanning(frame)))
        if prev is not None:
            fluxes.append(float(np.mean(np.maximum(0.0, mag - prev))))
        prev = mag
    return float(np.mean(fluxes)) if fluxes else float("nan")


def rt60_s(y: np.ndarray, sr: int) -> float:
    env = _envelope(y, sr)
    if env.size < 8:
        return float("nan")
    edb = 20 * np.log10(env / (env.max() + 1e-12) + 1e-12)
    valid = edb[edb < -5]
    if valid.size < 4:
        return 0.3
    t = np.arange(len(edb)) * (128 / sr)
    m = np.polyfit(t, edb, 1)[0]
    if m >= -1e-6:
        return 0.3
    return float(np.clip(-60.0 / m, 0.1, 5.0))


def drr_db(y: np.ndarray, sr: int) -> float:
    direct = int(0.04 * sr)
    reverb = int(min(0.5 * sr, len(y) - direct))
    if len(y) < direct + reverb:
        return float("nan")
    pd = float(np.sum(y[:direct] ** 2) + 1e-12)
    pr = float(np.sum(y[direct : direct + reverb] ** 2) + 1e-12)
    return float(np.clip(10 * np.log10(pd / pr), -20, 40))


def temporal_modulation(y: np.ndarray, sr: int) -> float:
    env = _envelope(y, sr)
    if env.size < 4:
        return float("nan")
    cv = float(env.std() / (env.mean() + 1e-9))
    return float(min(1.0, cv * 2.0))


METRIC_FN = {
    "attack_time": lambda y, sr: attack_time_ms(y, sr),
    "decay_rate": decay_rate,
    "f0": fundamental_frequency_hz,
    "spectral_centroid": spectral_centroid_hz,
    "spectral_flux": spectral_flux,
    "spectral_rolloff": spectral_rolloff_hz,
    "temporal_modulation": temporal_modulation,
    "rt60": rt60_s,
    "drr": drr_db,
}
