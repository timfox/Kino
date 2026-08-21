"""Dual timbre + tempo style encoders (§2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.audioim.config import AudioImConfig


def encode_timbre(waveform: np.ndarray, *, cfg: AudioImConfig | None = None) -> np.ndarray:
    """BEATs timbre stub — voiceprint-like embedding."""
    cfg = cfg or AudioImConfig()
    x = np.asarray(waveform, dtype=np.float64).ravel()
    if x.size == 0:
        x = np.zeros(4410)
    # Simple spectral envelope proxy
    spec = np.abs(np.fft.rfft(x))
    dim = 64
    idx = np.linspace(0, len(spec) - 1, dim).astype(int)
    return spec[idx] / (np.linalg.norm(spec[idx]) + 1e-8)


def encode_tempo(waveform: np.ndarray, *, cfg: AudioImConfig | None = None) -> np.ndarray:
    """Style Conditioner RVQ stub — rhythm/tempo embedding."""
    cfg = cfg or AudioImConfig()
    x = np.asarray(waveform, dtype=np.float64).ravel()
    frame = max(1, len(x) // cfg.tempo_codebooks)
    codes = []
    for i in range(cfg.tempo_codebooks):
        seg = x[i * frame : (i + 1) * frame]
        codes.append(float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0)
    return np.array(codes, dtype=np.float64)


def fuse_style(
    f_timbre: np.ndarray,
    f_tempo: np.ndarray,
) -> np.ndarray:
    """F_style = MLP(F_timbre + F_tempo) stub — concat + normalize."""
    fused = np.concatenate([f_timbre, f_tempo])
    return fused / (np.linalg.norm(fused) + 1e-8)


def style_conditioning_demo(*, seed: int = 0, cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    rng = np.random.default_rng(seed)
    wave = np.sin(2 * np.pi * 440 * np.linspace(0, cfg.prompt_seconds, int(44100 * cfg.prompt_seconds)))
    wave += 0.3 * rng.standard_normal(wave.shape)
    ft = encode_timbre(wave, cfg=cfg)
    fp = encode_tempo(wave, cfg=cfg)
    fs = fuse_style(ft, fp)
    return {
        "timbre_encoder": cfg.timbre_encoder,
        "tempo_encoder": cfg.tempo_encoder,
        "timbre_dim": len(ft),
        "tempo_dim": len(fp),
        "fused_dim": len(fs),
        "codebooks": cfg.tempo_codebooks,
    }
