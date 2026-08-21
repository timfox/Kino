"""Energy-based gating (Eq. 12–16, 29)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.mari.low_rank import LowRankAdapter


def layer_responses(
    h_base: np.ndarray,
    h_probe: np.ndarray,
) -> np.ndarray:
    """Per-layer L2 response ||h_probe - h_base||_2."""
    base = np.asarray(h_base, dtype=np.float64)
    probe = np.asarray(h_probe, dtype=np.float64)
    if base.shape != probe.shape:
        raise ValueError("base and probe layer stacks must match")
    return np.linalg.norm(probe - base, axis=-1)


def propagation_energy(
    h_base_layers: np.ndarray,
    h_probe_layers: np.ndarray,
) -> float:
    """Median propagation response E(x; α) (Eq. 12)."""
    em = layer_responses(h_base_layers, h_probe_layers)
    return float(np.median(em))


def simulate_post_injection_layers(
    h_inject: np.ndarray,
    delta: np.ndarray,
    *,
    num_layers: int,
    attenuation: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Toy forward: base layers decay identity; probe adds scaled delta with gain.

    Returns (base_layers, probe_layers) each (L, d).
    """
    h = np.asarray(h_inject, dtype=np.float64).reshape(-1)
    d = h.size
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    base = np.zeros((num_layers, d))
    probe = np.zeros((num_layers, d))
    cur_b = h.copy()
    cur_p = h + delta
    for i in range(num_layers):
        base[i] = cur_b
        probe[i] = cur_p
        cur_b = attenuation * cur_b
        cur_p = attenuation * cur_p
    return base, probe


def probe_delta(probe: LowRankAdapter, h: np.ndarray, *, alpha_probe: float) -> np.ndarray:
    return alpha_probe * probe.delta(h)


def calibrate_energy_threshold(
    energies: np.ndarray,
    applicable: np.ndarray,
    *,
    rho: float = 0.9,
) -> float:
    """
    Set τ_E as (1-ρ) quantile of non-applicable energies (Appendix A).

    applicable: bool array, True = should intervene.
    """
    energies = np.asarray(energies, dtype=np.float64)
    applicable = np.asarray(applicable, dtype=bool)
    non = energies[~applicable]
    if non.size == 0:
        return float(np.quantile(energies, 1.0 - rho))
    return float(np.quantile(non, 1.0 - rho))


@dataclass
class EnergyGate:
    threshold: float
    alpha_probe: float
    alpha_full: float
    alpha_safe: float

    def actuation_alpha(self, energy: float) -> float:
        if energy >= self.threshold:
            return self.alpha_full
        return self.alpha_safe

    def is_applicable(self, energy: float) -> bool:
        return energy >= self.threshold
