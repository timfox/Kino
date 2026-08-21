"""Numpy VAE encode/decode for HoliTok tokenizer (Stage II)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.vae import kl_gaussian, latent_sequence_shape, stage_ii_vae_loss


def _frame_wave(wave: np.ndarray, hop: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).ravel()
    n = max(1, (wave.size + hop - 1) // hop)
    frames = np.zeros((n, hop), dtype=np.float64)
    for i in range(n):
        chunk = wave[i * hop : (i + 1) * hop]
        if chunk.size < hop:
            chunk = np.pad(chunk, (0, hop - chunk.size))
        frames[i] = chunk
    return frames


def encode(wave: np.ndarray, *, cfg: HoliTokConfig | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic VAE encode: waveform → latent mean/logvar."""
    cfg = cfg or HoliTokConfig()
    hop = cfg.encoder_hop
    frames = _frame_wave(wave, hop)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((frames.shape[1], cfg.latent_dim)) * 0.02
    mean = frames @ proj
    log_var = np.log1p(np.var(frames, axis=1, keepdims=True) + 1e-6) * np.ones((mean.shape[0], cfg.latent_dim))
    return {"mean": mean, "log_var": log_var, "latent": mean}


def decode(latent: np.ndarray, *, cfg: HoliTokConfig | None = None, seed: int = 0) -> np.ndarray:
    cfg = cfg or HoliTokConfig()
    z = np.asarray(latent, dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    proj = rng.standard_normal((z.shape[-1], cfg.encoder_hop)) * 0.02
    wave = (z @ proj).reshape(-1)
    return wave


def encode_decode_smoke(cfg: HoliTokConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    sr = cfg.sample_rate_hz
    wave = np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr)
    enc = encode(wave, cfg=cfg)
    recon = decode(enc["latent"], cfg=cfg)
    loss = stage_ii_vae_loss(wave[: recon.size], recon, enc["mean"], enc["log_var"], cfg=cfg)
    shape = latent_sequence_shape(wave.size, cfg=cfg)
    return {
        "latent_shape": list(enc["latent"].shape),
        "expected_shape": list(shape),
        "kl_finite": np.isfinite(kl_gaussian(enc["mean"], enc["log_var"])),
        "stage_ii_total": loss["total"],
    }
