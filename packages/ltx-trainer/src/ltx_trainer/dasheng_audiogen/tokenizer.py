"""Dasheng 1.2B audio tokenizer proxy (1280-d @ 25 Hz)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig


def _frame_waveform(wave: np.ndarray, sample_rate: float, *, frame_hz: float) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).ravel()
    hop = max(1, int(sample_rate / frame_hz))
    n_frames = max(1, (wave.size + hop - 1) // hop)
    frames = np.zeros((n_frames, hop), dtype=np.float64)
    for i in range(n_frames):
        chunk = wave[i * hop : (i + 1) * hop]
        if chunk.size < hop:
            chunk = np.pad(chunk, (0, hop - chunk.size))
        frames[i] = chunk
    return frames


def encode_waveform(
    wave: np.ndarray,
    sample_rate: float,
    *,
    cfg: DashengAudioGenConfig | None = None,
) -> np.ndarray:
    """Project waveform frames to 1280-d latent tokens (deterministic hash embedding)."""
    cfg = cfg or DashengAudioGenConfig()
    frames = _frame_waveform(wave, sample_rate, frame_hz=cfg.latent_hz)
    # Mel-like stats per frame → linear projection to latent_dim
    mel_proxy = np.stack(
        [
            np.mean(frames, axis=1),
            np.std(frames, axis=1),
            np.max(np.abs(frames), axis=1),
            np.mean(np.abs(np.diff(frames, axis=1)), axis=1),
        ],
        axis=1,
    )
    proj = np.random.default_rng(42).standard_normal((mel_proxy.shape[1], cfg.latent_dim))
    z = mel_proxy @ proj
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z.astype(np.float64)


def decode_latents(z: np.ndarray, *, sample_rate: float = 48000.0, frame_hz: float = 25.0) -> np.ndarray:
    """Inverse proxy: latent energy → PCM envelope."""
    z = np.asarray(z, dtype=np.float64)
    energy = np.linalg.norm(z, axis=1)
    hop = max(1, int(sample_rate / frame_hz))
    wave = np.repeat(energy / (np.max(energy) + 1e-8), hop)
    return wave.astype(np.float64)


def tokenizer_smoke(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    sr = 48000.0
    wave = np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr)
    z = encode_waveform(wave, sr, cfg=cfg)
    recon = decode_latents(z, sample_rate=sr, frame_hz=cfg.latent_hz)
    return {
        "latent_shape": list(z.shape),
        "latent_dim": cfg.latent_dim,
        "latent_hz": cfg.latent_hz,
        "recon_samples": int(recon.size),
        "roundtrip_energy": float(np.sqrt(np.mean(recon**2))),
    }
