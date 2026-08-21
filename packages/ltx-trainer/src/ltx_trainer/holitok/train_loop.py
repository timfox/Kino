"""Stage II VAE training loop proxy (HoliTok §3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.encoder import decode, encode
from ltx_trainer.holitok.vae import stage_ii_vae_loss


def run_stage_ii_loop(
    wave: np.ndarray,
    *,
    steps: int = 12,
    lr: float = 0.05,
    cfg: HoliTokConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Toy latent nudge loop minimizing Stage II VAE loss."""
    cfg = cfg or HoliTokConfig()
    enc = encode(wave, cfg=cfg, seed=seed)
    latent = enc["latent"].copy()
    mean = enc["mean"].copy()
    log_var = enc["log_var"].copy()
    losses: list[float] = []

    for step in range(steps):
        recon = decode(latent, cfg=cfg, seed=seed + step)
        n = min(wave.size, recon.size)
        loss = stage_ii_vae_loss(wave[:n], recon[:n], mean, log_var, beta_low=cfg.beta_low, cfg=cfg)
        losses.append(loss["total"])
        # Finite-difference style nudge toward lower reconstruction error
        grad_proxy = (recon[:n] - wave[:n]).mean()
        latent -= lr * grad_proxy

    return {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] <= losses[0],
        "latent_frames": int(latent.shape[0]),
    }


def run_stage_iii_loop(
    wave: np.ndarray,
    *,
    steps: int = 10,
    lr: float = 0.03,
    cfg: HoliTokConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Stage III loop: nudge latents toward WavLM/x-vector teacher alignment."""
    from ltx_trainer.holitok.vae import stage_iii_loss

    cfg = cfg or HoliTokConfig()
    rng = np.random.default_rng(seed)
    enc = encode(wave, cfg=cfg, seed=seed)
    latent = enc["latent"].copy()
    mean = enc["mean"].copy()
    log_var = enc["log_var"].copy()
    d = cfg.latent_dim
    teacher_frame = rng.standard_normal((latent.shape[0], d * 2))
    teacher_utt = rng.standard_normal(d)
    losses: list[float] = []
    distill: list[float] = []

    for step in range(steps):
        recon = decode(latent, cfg=cfg, seed=seed + step)
        n = min(wave.size, recon.size)
        loss = stage_iii_loss(
            wave[:n],
            recon[:n],
            mean,
            log_var,
            teacher_frame,
            teacher_utt,
            latent,
            beta_high=cfg.beta_high,
            cfg=cfg,
        )
        losses.append(loss["total"])
        distill.append(loss["l_distill"])
        frame_tgt = teacher_frame[: latent.shape[0], : latent.shape[1]]
        latent -= lr * (latent - frame_tgt)

    return {
        "steps": steps,
        "initial_total": losses[0],
        "final_total": losses[-1],
        "initial_distill": distill[0],
        "final_distill": distill[-1],
        "distill_improved": distill[-1] <= distill[0],
        "latent_frames": int(latent.shape[0]),
    }


def train_loop_smoke(cfg: HoliTokConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    sr = cfg.sample_rate_hz
    wave = np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.5)) / sr)
    s2 = run_stage_ii_loop(wave, cfg=cfg, seed=seed)
    s3 = run_stage_iii_loop(wave, cfg=cfg, seed=seed + 1)
    return {
        "stage_ii_ran": True,
        "stage_iii_ran": True,
        "stage_ii_loss_decreased": s2["loss_decreased"],
        "stage_iii_distill_improved": s3["distill_improved"],
        **{f"stage_ii_{k}": v for k, v in s2.items() if k not in {"loss_decreased"}},
        **{f"stage_iii_{k}": v for k, v in s3.items() if k not in {"distill_improved"}},
    }
