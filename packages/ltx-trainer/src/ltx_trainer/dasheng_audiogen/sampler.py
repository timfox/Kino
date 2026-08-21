"""Euler flow-matching sampler with CFG (Dasheng AudioGen §3.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.captions import StructuredCaption
from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.model import FlowMatchingDiT
from ltx_trainer.dasheng_audiogen.tokenizer import decode_latents


def encode_text_views(caption: StructuredCaption, *, dim: int = 256) -> np.ndarray:
    """Bag-of-views text embedding for cross-attention."""
    rng = np.random.default_rng(hash(caption.caption) % 2**32)
    emb = np.zeros(dim, dtype=np.float64)
    for view, text in caption.to_sequence():
        h = abs(hash(view + text)) % (2**31)
        vec = rng.standard_normal(dim) * (0.5 + 0.1 * len(text))
        emb += vec * (h % 1000) / 1000.0
    return emb / (np.linalg.norm(emb) + 1e-8)


def sample_flow_matching(
    caption: StructuredCaption,
    *,
    duration_s: float = 10.0,
    cfg: DashengAudioGenConfig | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate latent trajectory z_0 → z_1 via Euler ODE with classifier-free guidance."""
    cfg = cfg or DashengAudioGenConfig()
    steps = steps or cfg.fm_steps
    guidance = guidance if guidance is not None else cfg.cfg_scale
    rng = np.random.default_rng(seed)
    T = int(cfg.latent_hz * duration_s)
    d = min(64, cfg.latent_dim)
    z = rng.standard_normal((T, d))
    text = encode_text_views(caption, dim=cfg.text_dim)
    null = np.zeros_like(text)
    model = FlowMatchingDiT(latent_dim=d, text_dim=cfg.text_dim, rng=rng)
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v_cond = model.predict_velocity(z, t, text)
        v_uncond = model.predict_velocity(z, t, null)
        v = v_uncond + guidance * (v_cond - v_uncond)
        z = z + dt * v
    return z


def sample_to_waveform(
    caption: StructuredCaption,
    *,
    sample_rate: float = 48000.0,
    **kwargs: Any,
) -> np.ndarray:
    cfg = kwargs.pop("cfg", None) or DashengAudioGenConfig()
    z = sample_flow_matching(caption, cfg=cfg, **kwargs)
    return decode_latents(z, sample_rate=sample_rate, frame_hz=cfg.latent_hz)


def sampler_smoke(cfg: DashengAudioGenConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    cap = StructuredCaption(
        caption="Synth pop with female German speech.",
        speech="Female voice in German.",
        music="Processed guitar pop bed.",
    )
    z = sample_flow_matching(cap, duration_s=2.0, cfg=cfg, seed=seed)
    wave = decode_latents(z, sample_rate=48000.0, frame_hz=cfg.latent_hz)
    return {
        "latent_frames": int(z.shape[0]),
        "wave_samples": int(wave.size),
        "cfg_scale": cfg.cfg_scale,
        "fm_steps": cfg.fm_steps,
    }
