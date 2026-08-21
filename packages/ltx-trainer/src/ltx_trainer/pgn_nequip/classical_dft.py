"""Classical DFT free-energy stub for solvent-free PGNs (Eqs. 1–4)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.pgn_nequip.config import PGNDesign


def grafting_count(design: PGNDesign) -> float:
    """Eq. S1 — grafted chains per particle."""
    na = 6.022e23
    v_core = (np.pi / 6.0) * (design.core_diameter_nm * 1e-9) ** 3
    void_per_core = ((1.0 - design.phi_c) / max(design.phi_c, 1e-9)) * v_core
    vol_per_chain = (design.mw_kda * 1000.0) / (design.rho_p_g_cm3 * 1000.0 * na)
    return void_per_core * design.rho_p_g_cm3 * 1000.0 * na / (design.mw_kda * 1000.0)


def polymer_density_profile_stub(
    positions: np.ndarray,
    *,
    design: PGNDesign,
    alpha: float = 1e6,
) -> np.ndarray:
    """Uniform-density target for solvent-free limit (Fig. 2–3)."""
    _ = alpha
    n = positions.shape[0]
    # Heterogeneity proxy: Voronoi-like volume spread raises energy.
    com = positions.mean(axis=0)
    radii = np.linalg.norm(positions - com, axis=1)
    uniform = np.full(n, radii.mean())
    return uniform / (uniform.sum() + 1e-12)


def energy_per_particle_kbt(
    positions: np.ndarray,
    *,
    design: PGNDesign,
    alpha: float = 1e6,
) -> float:
    """Many-body polymer free energy proxy (Eq. 1) in kBT units."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    # Pair distances (cores as unit diameter).
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + np.eye(n)
    min_dist = dist.min(axis=1)

    overlap_penalty = np.sum(np.exp(-2.0 * (min_dist - 1.0)))
    com = positions.mean(axis=0)
    spread = np.linalg.norm(positions - com, axis=1)
    void_heterogeneity = np.var(spread) * (1.0 / max(design.phi_c, 0.05))

    # Entropic frustration grows at low phi_c and high grafting density.
    frustration = (1.0 / max(design.phi_c, 0.05)) * (design.sigma_g_chains_nm2 / 1.8)

    raw = 0.5 * overlap_penalty + alpha * 1e-6 * void_heterogeneity + 0.15 * frustration
    return float(raw / n)


def energy_distribution_stats(
    energies_kbt: np.ndarray,
) -> dict[str, float]:
    e = np.asarray(energies_kbt, dtype=np.float64)
    return {
        "mean_kbt": float(np.mean(e)),
        "std_kbt": float(np.std(e)),
        "max_kbt": float(np.max(e)),
    }
