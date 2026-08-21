"""Trigger-free watermark embedding objectives (Cert-LAS §4.4)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cert_las.config import CertLASConfig


def target_distribution(cfg: CertLASConfig | None = None) -> np.ndarray:
    cfg = cfg or CertLASConfig()
    lam = float(cfg.target_lambda)
    return np.array([1.0 - lam, lam], dtype=np.float64)


def kl_watermark_loss(
    classifier_probs: np.ndarray,
    *,
    cfg: CertLASConfig | None = None,
) -> float:
    """LKL: D_KL(q* || p_phi(.|x)) for binary prompts (Eq. 2)."""
    q = target_distribution(cfg)
    p = np.clip(classifier_probs, 1e-8, 1.0)
    p = p / p.sum()
    return float(np.sum(q * (np.log(q + 1e-12) - np.log(p))))


def ms_ssim_stub(x: np.ndarray, y: np.ndarray) -> float:
    """MS-SSIM surrogate in [0,1] for perceptual regularizer (Eq. 3)."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(np.clip(num / max(den, 1e-12), 0.0, 1.0))


def fidelity_loss(x_theta: np.ndarray, x_phi: np.ndarray) -> float:
    return 1.0 - ms_ssim_stub(x_theta, x_phi)


def exponential_growth_schedule(step: int, *, Tg: int, m_max: int, m_min: int = 1) -> tuple[int, float]:
    """m_t = min(m_max, floor(2^{t/Tg})); ω_t doubles each period (stub ω scale)."""
    Tg = max(1, int(Tg))
    m_t = min(m_max, max(m_min, int(2 ** (step / Tg))))
    omega_scale = 2.0 ** (step / Tg)
    return m_t, omega_scale
