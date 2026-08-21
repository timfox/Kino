"""Wavelet-inspired parent-child refinement (Options A–D, Eq. 3–4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.evogs.config import RefinementMode


def expand_alpha(alpha: np.ndarray, dim: int) -> np.ndarray:
    """Broadcast 5-group asymmetry vector to full parameter dimension."""
    if alpha.shape[0] >= dim:
        return alpha[:dim]
    groups = np.array_split(np.arange(dim), alpha.shape[0])
    out = np.ones(dim, dtype=np.float64)
    for g, a in zip(groups, alpha, strict=False):
        out[g] = float(a)
    return out


def split_children(
    parent: np.ndarray,
    psi: np.ndarray,
    alpha: np.ndarray,
    *,
    mode: RefinementMode = RefinementMode.ASYMMETRIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce child splats from parent P and refinement (ψ, α)."""
    p = np.asarray(parent, dtype=np.float64)
    w = np.asarray(psi, dtype=np.float64)
    if mode == RefinementMode.INDEPENDENT:
        rng = np.random.default_rng(int(abs(p.sum() * 1e4)) % 2**31)
        c1 = p + rng.normal(scale=0.1, size=p.shape)
        c2 = p + rng.normal(scale=0.1, size=p.shape)
        return c1, c2
    if mode == RefinementMode.INDEP_RESIDUAL:
        rng = np.random.default_rng(int(abs(p.sum() * 1e4)) % 2**31)
        d1 = rng.normal(scale=0.05, size=p.shape)
        d2 = rng.normal(scale=0.05, size=p.shape)
        return p + d1, p + d2
    if mode == RefinementMode.SYMMETRIC:
        return p + w, p - w
    a = expand_alpha(np.asarray(alpha, dtype=np.float64), p.shape[0])
    return p + w, p - a * w


def child_mean(parent: np.ndarray, psi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(C1 + C2) / 2 for asymmetric Option D."""
    c1, c2 = split_children(parent, psi, alpha, mode=RefinementMode.ASYMMETRIC)
    return 0.5 * (c1 + c2)


def reconstruct_leaf(
    root: np.ndarray,
    branch_signs: list[float],
    psis: list[np.ndarray],
    alphas: list[np.ndarray],
) -> np.ndarray:
    """
    Accumulate refinements along ancestral chain (Sec. 3.1).

    S = P_root + Σ s_k ⊙ ψ_k with s_k ∈ {+1, −α_k} on the taken branch.
    """
    out = np.asarray(root, dtype=np.float64).copy()
    for sign, psi, alpha in zip(branch_signs, psis, alphas, strict=False):
        if sign >= 0:
            out = out + psi
        else:
            out = out - expand_alpha(alpha, out.shape[0]) * psi
    return out


def psi_energy_concentration(psi: np.ndarray, top_fraction: float = 0.2) -> float:
    """Fraction of ||ψ||² energy in top ``top_fraction`` coefficients (Fig. 5)."""
    flat = np.abs(psi.ravel())
    if flat.size == 0:
        return 0.0
    order = np.argsort(flat)[::-1]
    k = max(1, int(round(top_fraction * flat.size)))
    total = float(np.sum(flat**2)) + 1e-12
    top = float(np.sum(flat[order[:k]] ** 2))
    return top / total


def refinement_summary(psi: np.ndarray) -> dict[str, Any]:
    return {
        "psi_l2": float(np.linalg.norm(psi)),
        "psi_max_abs": float(np.max(np.abs(psi))),
        "energy_top20pct": psi_energy_concentration(psi, 0.2),
    }
