"""Mel-spectrogram CNN baseline params (§4.5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mygardenbird.config import MygardenbirdConfig


def mel_params(cfg: MygardenbirdConfig | None = None, *, rate: int | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    sr = rate or c.sample_rate_16k
    n_samples = sr * int(c.clip_duration_s)
    n_fft = 2048
    n_mels = 224
    hop = n_samples // 224
    return {
        "sample_rate": sr,
        "clip_duration_s": c.clip_duration_s,
        "n_samples": n_samples,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop,
        "target_frames": 224,
        "image_size": (224, 224),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "mixup_alpha": 0.2,
    }


def spectrogram_demo(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    p16 = mel_params(c, rate=c.sample_rate_16k)
    p44 = mel_params(c, rate=c.sample_rate_44k)
    return {
        "16kHz": p16,
        "44kHz": p44,
        "hop_increases_at_44k": p44["hop_length"] > p16["hop_length"],
    }
