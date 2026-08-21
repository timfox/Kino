"""Gradient preconditioner (GP) stubs paired with AP in DoPr."""

from __future__ import annotations

import numpy as np

from ltx_trainer.dopr.config import GpKind


def _signum(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def _adam_like(
    m: np.ndarray,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    state: dict | None = None,
    step: int = 1,
) -> tuple[np.ndarray, dict]:
    state = state or {}
    m1 = beta1 * state.get("m1", np.zeros_like(m)) + (1.0 - beta1) * m
    m2 = beta2 * state.get("m2", np.zeros_like(m)) + (1.0 - beta2) * (m * m)
    m1_hat = m1 / (1.0 - beta1**step)
    m2_hat = m2 / (1.0 - beta2**step)
    update = m1_hat / (np.sqrt(m2_hat) + eps)
    return update, {"m1": m1, "m2": m2}


def _muon(m: np.ndarray) -> np.ndarray:
    """Orthonormalize via SVD (small-matrix Muon stub)."""
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def gradient_precondition(
    m: np.ndarray,
    gp: GpKind = "adamw",
    *,
    state: dict | None = None,
    step: int = 1,
) -> tuple[np.ndarray, dict]:
    if gp in ("sgd",):
        return m.copy(), state or {}
    if gp in ("signum",):
        return _signum(m), state or {}
    if gp in ("adam", "adamw", "adamuon"):
        return _adam_like(m, state=state, step=step)
    if gp == "muon":
        return _muon(m), state or {}
    raise ValueError(f"unsupported gp: {gp}")
