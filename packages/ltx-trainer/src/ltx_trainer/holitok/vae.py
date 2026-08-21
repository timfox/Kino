"""HoliTok VAE tokenizer math — compression, KL, staged losses (numpy)."""

from __future__ import annotations

import math

import numpy as np

from ltx_trainer.holitok.config import HoliTokConfig


def compression_ratio(
    sample_rate_hz: int,
    latent_frame_rate_hz: float,
    latent_dim: int,
    *,
    bfloat_bits: int = 32,
) -> float:
    """Eq. 13 — nominal bitrate compression vs raw PCM."""
    bits_per_sample = math.ceil(math.log2(max(sample_rate_hz, 2)))
    waveform_bps = sample_rate_hz * bits_per_sample
    latent_bps = latent_frame_rate_hz * latent_dim * bfloat_bits
    return waveform_bps / latent_bps


def kl_gaussian(mean: np.ndarray, log_var: np.ndarray) -> float:
    """Diagonal Gaussian KL to N(0, I), averaged over elements."""
    mean = np.asarray(mean, dtype=np.float64)
    log_var = np.asarray(log_var, dtype=np.float64)
    kl = -0.5 * (1.0 + log_var - mean**2 - np.exp(log_var))
    return float(np.mean(kl))


def generator_loss(
    waveform: np.ndarray,
    reconstruction: np.ndarray,
    *,
    lambda_spec: float = 45.0,
    lambda_adv: float = 1.0,
    lambda_fm: float = 2.0,
) -> float:
    """Stage I ℓ_gen proxy: spectral + adversarial + feature-matching surrogates."""
    w = np.asarray(waveform, dtype=np.float64).ravel()
    r = np.asarray(reconstruction, dtype=np.float64).ravel()
    n = min(w.size, r.size)
    w, r = w[:n], r[:n]
    l_spec = float(np.mean((w - r) ** 2))
    l_adv = float(np.mean(np.abs(w - r)))
    l_fm = float(np.std(w - r))
    return lambda_spec * l_spec + lambda_adv * l_adv + lambda_fm * l_fm


def stage_ii_vae_loss(
    waveform: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    log_var: np.ndarray,
    *,
    beta_low: float = 0.1,
    cfg: HoliTokConfig | None = None,
) -> dict[str, float]:
    """Stage II reconstruction-dominated VAE objective (Eq. 3)."""
    c = cfg or HoliTokConfig()
    l_gen = generator_loss(waveform, reconstruction, lambda_spec=c.lambda_spec, lambda_adv=c.lambda_adv, lambda_fm=c.lambda_fm)
    l_kl = kl_gaussian(mean, log_var)
    total = l_gen + beta_low * l_kl
    return {"l_gen": l_gen, "l_kl": l_kl, "total": total}


def cosine_distill_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """1 − cos similarity (Eq. 4 frame/utterance distillation)."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    n = min(p.size, t.size)
    p, t = p[:n], t[:n]
    denom = (np.linalg.norm(p) * np.linalg.norm(t)) + 1e-8
    cos = float(np.dot(p, t) / denom)
    return 1.0 - cos


def stage_iii_loss(
    waveform: np.ndarray,
    reconstruction: np.ndarray,
    mean: np.ndarray,
    log_var: np.ndarray,
    teacher_frame: np.ndarray,
    teacher_utt: np.ndarray,
    latent_pred: np.ndarray,
    *,
    beta_high: float = 7.0,
    cfg: HoliTokConfig | None = None,
) -> dict[str, float]:
    """Stage III downstream-aware enrichment (Eq. 6)."""
    c = cfg or HoliTokConfig()
    l_gen = generator_loss(waveform, reconstruction, lambda_spec=c.lambda_spec, lambda_adv=c.lambda_adv, lambda_fm=c.lambda_fm)
    l_kl = kl_gaussian(mean, log_var)
    l_wavlm = cosine_distill_loss(latent_pred, teacher_frame)
    l_xvec = cosine_distill_loss(np.mean(latent_pred, axis=0) if latent_pred.ndim > 1 else latent_pred, teacher_utt)
    l_distill = c.lambda_wavlm_distill * l_wavlm + c.lambda_xvector_distill * l_xvec
    # Supervision CE proxy: negative log prob of random target token
    logits = np.tanh(latent_pred.ravel()[:8]) if latent_pred.size else np.array([0.0])
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum() + 1e-8
    l_sup = float(-np.log(probs[0] + 1e-8))
    total = l_gen + beta_high * l_kl + l_distill + c.lambda_sup * l_sup
    return {
        "l_gen": l_gen,
        "l_kl": l_kl,
        "l_distill": l_distill,
        "l_sup": l_sup,
        "total": total,
    }


def implicit_fidelity_bound(epsilon_ae: float, delta_shift: float, lipschitz: float = 1.0) -> float:
    """Prop. 1 upper bound on variational reconstruction distortion."""
    return 2.0 * epsilon_ae + 2.0 * (lipschitz**2) * delta_shift


def latent_sequence_shape(num_samples: int, *, cfg: HoliTokConfig | None = None) -> tuple[int, int]:
    """Return (T, D) latent shape for a given waveform length in samples."""
    c = cfg or HoliTokConfig()
    frames = max(1, int(math.ceil(num_samples / c.encoder_hop)))
    return frames, c.latent_dim
