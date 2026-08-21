"""Complete-view State Anchoring losses (Sec. 4.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.core_kd.config import CoreKDConfig
from ltx_trainer.core_kd.gaussian_state import dg_distance, kl_distill_loss


def cross_entropy(logits: np.ndarray, label: int) -> float:
    z = np.asarray(logits, dtype=np.float64).ravel()
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(-np.log(p[label] + 1e-9))


def prediction_anchoring_loss(
    y: int,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    cfg: CoreKDConfig | None = None,
) -> float:
    cfg = cfg or CoreKDConfig()
    ce = cross_entropy(student_logits, y)
    kd = kl_distill_loss(teacher_logits, student_logits, temperature=cfg.kd_temperature)
    return ce + cfg.lambda_kd * kd


def fused_state_loss(
    student_mu: np.ndarray,
    student_sigma: np.ndarray,
    teacher_mu: np.ndarray,
    teacher_sigma: np.ndarray,
) -> float:
    return dg_distance(student_mu, student_sigma, teacher_mu, teacher_sigma)


def unavailable_modality_state_loss(
    predicted_states: dict[str, tuple[np.ndarray, np.ndarray]],
    teacher_states: dict[str, tuple[np.ndarray, np.ndarray]],
    unavailable: set[str],
) -> float:
    if not unavailable:
        return 0.0
    dists: list[float] = []
    for m in unavailable:
        if m not in predicted_states or m not in teacher_states:
            continue
        smu, ssig = predicted_states[m]
        tmu, tsig = teacher_states[m]
        dists.append(dg_distance(smu, ssig, tmu, tsig))
    return float(np.mean(dists)) if dists else 0.0


def csa_loss(
    y: int,
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    student_fused: tuple[np.ndarray, np.ndarray],
    teacher_fused: tuple[np.ndarray, np.ndarray],
    predicted_unavail: dict[str, tuple[np.ndarray, np.ndarray]],
    teacher_modal: dict[str, tuple[np.ndarray, np.ndarray]],
    unavailable: set[str],
    *,
    cfg: CoreKDConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or CoreKDConfig()
    l_pred = prediction_anchoring_loss(y, student_logits, teacher_logits, cfg=cfg)
    l_state = fused_state_loss(*student_fused, *teacher_fused)
    l_mstate = unavailable_modality_state_loss(predicted_unavail, teacher_modal, unavailable)
    total = l_pred + cfg.lambda_state * l_state + cfg.lambda_mstate * l_mstate
    return {
        "l_pred": l_pred,
        "l_state": l_state,
        "l_mstate": l_mstate,
        "l_csa": total,
    }
