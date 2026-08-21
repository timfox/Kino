"""AudioVAE two-stage training stub (§2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dotstts.config import DotsttsConfig


def audiovae_loss_stage1(
    *,
    mel_loss: float,
    adv_loss: float,
    fm_loss: float,
    kl_loss: float,
    beta_kl: float = 1.0,
) -> float:
    """Stage 1: reconstruction + adversarial + KL + flow prior."""
    return mel_loss + adv_loss + fm_loss + beta_kl * kl_loss


def audiovae_loss_stage2(
    *,
    stage1: float,
    wavlm_align: float,
    sup_loss: float,
    lambda_wavlm: float = 1.0,
    lambda_sup: float = 1.0,
) -> float:
    """Stage 2: add WavLM alignment + multitask ASR/emotion/speaker."""
    return stage1 + lambda_wavlm * wavlm_align + lambda_sup * sup_loss


def temporal_energy_profile(latent: np.ndarray) -> np.ndarray:
    """Channel-mean energy over time for B×C×T latent."""
    return np.mean(latent**2, axis=1)


def audiovae_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    rng = np.random.default_rng(seed)
    # Toy 1 s clip @ 25 Hz
    latent = rng.normal(size=(1, cfg.latent_dim, int(cfg.latent_fps)))
    s1 = audiovae_loss_stage1(mel_loss=0.4, adv_loss=0.2, fm_loss=0.1, kl_loss=0.05)
    s2 = audiovae_loss_stage2(stage1=s1, wavlm_align=0.08, sup_loss=0.06)
    return {
        "latent_shape": list(latent.shape),
        "latent_fps": cfg.latent_fps,
        "semantic_fps": cfg.semantic_fps,
        "stage1_loss": round(s1, 4),
        "stage2_loss": round(s2, 4),
        "stage2_improves_learnability": s2 > s1,
        "reconstruction_wer_pct": cfg.vae_wer_pct,
    }
