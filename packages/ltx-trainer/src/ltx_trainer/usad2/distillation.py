"""Domain-aware distillation and layer-wise loss stubs (§2.1–2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.usad2.config import Usad2Config


def domain_aware_weights(
    data_domain: int,
    *,
    num_teachers: int = 3,
    alpha: float = 10.0,
) -> np.ndarray:
    """
    Eq. (3): matched teacher gets α/(α+M−1), others get 1/(α+M−1).
    """
    m = num_teachers
    denom = alpha + m - 1
    weights = np.full(m, 1.0 / denom, dtype=np.float64)
    if 0 <= data_domain < m:
        weights[data_domain] = alpha / denom
    return weights


def usad_loss(
    teacher_losses: np.ndarray,
    data_domain: int,
    *,
    alpha: float = 10.0,
) -> float:
    """Eq. (2): L_USAD2 = Σ_m w_m(m_data) L_m."""
    w = domain_aware_weights(data_domain, num_teachers=len(teacher_losses), alpha=alpha)
    return float(np.dot(w, teacher_losses))


def layerwise_distill_stub(
    student: np.ndarray,
    teacher: np.ndarray,
    *,
    temperature: float = 0.1,
) -> float:
    """DistilHuBERT-style cosine similarity loss (minimize negative cosine)."""
    s = student / (np.linalg.norm(student) + 1e-8)
    t = teacher / (np.linalg.norm(teacher) + 1e-8)
    return float(1.0 - np.dot(s, t))


def distillation_demo(seed: int = 0, cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    rng = np.random.default_rng(seed)
    k_layers = 4
    losses = rng.uniform(0.2, 0.8, size=(c.num_teachers_ssl, k_layers))

    uniform = float(losses.mean())
    speech_weighted = usad_loss(losses.mean(axis=1), data_domain=0, alpha=c.domain_aware_alpha)
    music_weighted = usad_loss(losses.mean(axis=1), data_domain=2, alpha=c.domain_aware_alpha)

    w_speech = domain_aware_weights(0, num_teachers=c.num_teachers_ssl, alpha=c.domain_aware_alpha)
    w_music = domain_aware_weights(2, num_teachers=c.num_teachers_ssl, alpha=c.domain_aware_alpha)

    return {
        "alpha": c.domain_aware_alpha,
        "num_teachers": c.num_teachers_ssl,
        "weights_speech_domain": w_speech.tolist(),
        "weights_music_domain": w_music.tolist(),
        "matched_weight_ratio": float(w_speech[0] / w_speech[1]),
        "uniform_loss": uniform,
        "speech_domain_loss": speech_weighted,
        "music_domain_loss": music_weighted,
        "domain_aware_changes_loss": abs(speech_weighted - music_weighted) > 1e-6,
    }
