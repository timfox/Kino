"""Leaky Integrate-and-Fire neuron (Eq. 27–29)."""

from __future__ import annotations

import numpy as np


def lif_step(
    x: np.ndarray,
    membrane: np.ndarray,
    *,
    tau: float = 2.0,
    beta: float = 0.5,
    v_thr: float = 1.0,
    v_reset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One discrete LIF step; returns spikes and updated membrane."""
    h = membrane + (1.0 / tau) * (x - membrane - v_reset)
    spikes = (h >= v_thr).astype(np.float64)
    membrane_out = beta * h * (1.0 - spikes) + v_reset * spikes
    return spikes, membrane_out


def gradient_proxy(x: np.ndarray, *, alpha: float = 10.0) -> np.ndarray:
    """Sigmoid surrogate gradient σ(x) for SNN backprop (Eq. 41)."""
    return 1.0 / (1.0 + np.exp(-alpha * x))
