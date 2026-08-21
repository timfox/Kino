"""Domain generalization stubs: Fish, Fishr, GGA-L."""

from __future__ import annotations

from enum import Enum

import numpy as np


class DgMethod(str, Enum):
    FISH = "fish"
    FISHR = "fishr"
    GGA_L = "gga_l"


def gradient_cosine(g1: np.ndarray, g2: np.ndarray, eps: float = 1e-8) -> float:
    n1 = np.linalg.norm(g1) + eps
    n2 = np.linalg.norm(g2) + eps
    return float(np.dot(g1, g2) / (n1 * n2))


def fish_alignment(grads: list[np.ndarray]) -> float:
    """Mean pairwise gradient cosine (Shi et al., 2021)."""
    if len(grads) < 2:
        return 1.0
    flat = [g.ravel() for g in grads]
    scores = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            scores.append(gradient_cosine(flat[i], flat[j]))
    return float(np.mean(scores))


def fishr_variance_penalty(grads: list[np.ndarray]) -> float:
    """Variance of domain gradient norms (Rame et al., 2022)."""
    norms = np.array([np.linalg.norm(g) for g in grads])
    return float(np.var(norms))


def dg_objective(method: DgMethod, domain_grads: list[np.ndarray]) -> dict[str, float]:
    align = fish_alignment(domain_grads)
    var_pen = fishr_variance_penalty(domain_grads)
    if method == DgMethod.FISH:
        return {"alignment": align, "loss_term": -align}
    if method == DgMethod.FISHR:
        return {"alignment": align, "variance_penalty": var_pen, "loss_term": var_pen}
    # GGA-L: early gradient alignment proxy
    return {"alignment": align, "loss_term": -0.5 * align}
