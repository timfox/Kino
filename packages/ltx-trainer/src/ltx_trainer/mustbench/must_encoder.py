"""MERT-style encoder + MUST tokenization proxy (Sec. 4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mustbench.config import MustBenchConfig


def _mel_frames(wave: np.ndarray, sample_rate: float, *, hop_ms: float = 20.0, n_mels: int = 64) -> np.ndarray:
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


def encode_audio(wave: np.ndarray, *, cfg: MustBenchConfig | None = None) -> np.ndarray:
    """Waveform → MUST tokens at cfg.must_token_rate Hz."""
    cfg = cfg or MustBenchConfig()
    mel = _mel_frames(wave, cfg.sample_rate_hz)
    mel_frame_rate_hz = 1000.0 / 20.0
    frame_hop = max(1, int(round(mel_frame_rate_hz / cfg.must_token_rate)))
    tokens: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((mel.shape[0], cfg.must_token_dim))
    for i in range(0, mel.shape[1], frame_hop):
        seg = mel[:, i : i + frame_hop].mean(axis=1)
        tok = seg @ proj
        tokens.append(tok / (np.linalg.norm(tok) + 1e-8))
    return np.stack(tokens, axis=0) if tokens else np.zeros((1, cfg.must_token_dim))


def transition_probability(tokens: np.ndarray) -> np.ndarray:
    """Frame-wise transition probability from token delta energy."""
    z = np.asarray(tokens, dtype=np.float64)
    if z.shape[0] < 2:
        return np.zeros(z.shape[0])
    delta = np.linalg.norm(np.diff(z, axis=0), axis=1)
    delta = np.concatenate([[delta[0]], delta])
    p = delta / (delta.max() + 1e-8)
    return p.astype(np.float64)


def must_encoder_smoke(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    sr = cfg.sample_rate_hz
    t = np.arange(int(sr * 30)) / sr
    wave = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * (t > 15))
    tok = encode_audio(wave, cfg=cfg)
    trans = transition_probability(tok)
    return {"n_tokens": int(tok.shape[0]), "token_dim": cfg.must_token_dim, "transition_peak": float(trans.max())}
