"""On-policy privileged distillation helpers (reverse KL, Eq. 4 toy)."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def reverse_kl(student: np.ndarray, teacher: np.ndarray, *, eps: float = 1e-8) -> float:
    """DKL(p_student || p_teacher) for discrete distributions."""
    p = np.clip(student, eps, 1.0)
    q = np.clip(teacher, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def privileged_distillation_loss(
    student_logits_per_scale: list[np.ndarray],
    teacher_logits_per_scale: list[np.ndarray],
) -> float:
    """Average reverse KL across acceleration scales."""
    if len(student_logits_per_scale) != len(teacher_logits_per_scale):
        raise ValueError("scale count mismatch")
    losses = [
        reverse_kl(softmax(s), softmax(t))
        for s, t in zip(student_logits_per_scale, teacher_logits_per_scale, strict=True)
    ]
    return float(np.mean(losses))
