"""Waveform → layer hidden proxy for ChildVox weighted encoder pool."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.childvox.config import ChildVoxConfig


def _mel_frames(wave: np.ndarray, sample_rate: float, *, n_mels: int = 64, hop_ms: float = 10.0) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).ravel()
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    n_frames = max(1, wave.size // hop)
    mel = np.zeros((n_mels, n_frames), dtype=np.float64)
    for i in range(n_frames):
        chunk = wave[i * hop : (i + 1) * hop]
        if chunk.size < hop:
            chunk = np.pad(chunk, (0, hop - chunk.size))
        spec = np.abs(np.fft.rfft(chunk))
        bins = np.linspace(0, spec.size, n_mels + 1, dtype=int)
        for m in range(n_mels):
            seg = spec[bins[m] : bins[m + 1]]
            mel[m, i] = np.log1p(np.mean(seg) if seg.size else 0.0)
    return mel


def layer_hiddens_from_waveform(
    wave: np.ndarray,
    *,
    sample_rate: float | None = None,
    num_layers: int = 6,
    hidden_dim: int = 64,
) -> np.ndarray:
    """Build (layers, time, dim) tensor from mel statistics (deterministic)."""
    cfg = ChildVoxConfig()
    sr = sample_rate or cfg.sample_rate_hz
    mel = _mel_frames(wave, sr)
    t = mel.shape[1]
    layers: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((num_layers, mel.shape[0], hidden_dim)) * 0.05
    for li in range(num_layers):
        # deeper layers see progressively smoothed mel
        smooth = mel
        for _ in range(li // 2):
            smooth = np.pad(smooth, ((0, 0), (1, 1)), mode="edge")
            smooth = 0.25 * smooth[:, :-2] + 0.5 * smooth[:, 1:-1] + 0.25 * smooth[:, 2:]
        frame = (proj[li] @ smooth).T  # time × dim
        layers.append(frame)
    return np.stack(layers, axis=0)


def encoder_proxy_smoke(*, seed: int = 0) -> dict[str, Any]:
    sr = 16000.0
    t = np.arange(int(sr)) / sr
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    h = layer_hiddens_from_waveform(wave, sample_rate=sr)
    return {"shape": list(h.shape), "finite": bool(np.all(np.isfinite(h)))}
