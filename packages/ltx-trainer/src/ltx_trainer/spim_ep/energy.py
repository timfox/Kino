"""EP energy functional and SPIM Hamiltonian (Eqs. 1–4, 10)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.spim_ep.mattis import MattisParams, build_augmented_j, coupling_from_mattis
from ltx_trainer.spim_ep.nonlinear import rho, rho_prime


def interaction_energy(s: NDArray[np.floating], j_dyn: NDArray[np.floating]) -> float:
    """I(s) = −½ ρ(s)^T J_dyn ρ(s) (Eq. 2)."""
    ps = rho(s)
    return float(-0.5 * ps @ j_dyn @ ps)


def input_bias(s: NDArray[np.floating], u: NDArray[np.floating], j_in: NDArray[np.floating]) -> float:
    """B(s, u) = −ρ(u)^T J_in ρ(s) (Eq. 1)."""
    return float(-rho(u) @ j_in @ rho(s))


def total_energy(
    s: NDArray[np.floating],
    u: NDArray[np.floating],
    params: MattisParams,
    *,
    n_input: int,
    alpha: float,
    beta: float,
    y: NDArray[np.floating] | None,
    j: NDArray[np.floating] | None = None,
) -> float:
    """E = B + I + α‖s‖²/2 + β‖s_out − y‖²/2 (Eq. 1)."""
    if j is None:
        j = build_augmented_j(params.lambdas, params.xi, n_input)
    j_in = j[:n_input, n_input:]
    j_dyn = j[n_input:, n_input:]
    n_out = y.shape[0] if y is not None else 0
    s_out = s[-n_out:] if n_out else np.array([], dtype=s.dtype)
    e = input_bias(s, u, j_in) + interaction_energy(s, j_dyn) + 0.5 * alpha * float(np.sum(s * s))
    if y is not None and beta != 0.0 and n_out:
        e += 0.5 * beta * float(np.sum((s_out - y) ** 2))
    return e


def spim_hamiltonian(s: NDArray[np.floating], j: NDArray[np.floating]) -> float:
    """H_SPIM ∝ −Σ_ij J_ij sin(s_i) sin(s_j) (Eq. 4); uses ρ=sin on dynamic block."""
    ps = rho(s)
    return float(-0.5 * ps @ j @ ps)


def grad_sm_spim_finite_diff(
    s: NDArray[np.floating],
    j: NDArray[np.floating],
    m: int,
    delta: float,
) -> float:
    """∂H/∂s_m via ±δ phase shift (Eq. 6)."""
    sp = s.copy()
    sm = s.copy()
    sp[m] += delta
    sm[m] -= delta
    return spim_hamiltonian(sp, j) - spim_hamiltonian(sm, j)


def grad_energy_wrt_sm(
    s: NDArray[np.floating],
    u: NDArray[np.floating],
    params: MattisParams,
    m: int,
    *,
    n_input: int,
    alpha: float,
    beta: float,
    y: NDArray[np.floating] | None,
    j: NDArray[np.floating] | None = None,
    j_tilde: NDArray[np.floating] | None = None,
    delta: float,
    hybrid_digital_input: bool = False,
) -> float:
    """Combined ∂E/∂s_m per Eq. (10): SPIM interaction + digital bias + αs + β nudge."""
    if j is None:
        j = build_augmented_j(params.lambdas, params.xi, n_input)
    j_dyn = j[n_input:, n_input:]
    j_in = j[:n_input, n_input:]
    j_use = j_tilde if j_tilde is not None else j_dyn
    g_spim = grad_sm_spim_finite_diff(s, j_use, m, delta)
    g = g_spim
    if not hybrid_digital_input:
        g += -float((j_in.T @ rho(u))[m]) * float(rho_prime(s)[m])
    g += alpha * float(s[m])
    if y is not None and beta != 0.0:
        n_out = y.shape[0]
        if m >= s.shape[0] - n_out:
            g += beta * float(s[m] - y[m - (s.shape[0] - n_out)])
    return g


def grad_lambda_k(
    s: NDArray[np.floating],
    xi_k: NDArray[np.floating],
    rank: int,
    n_input: int,
) -> float:
    """∂I/∂λ_k ≈ −1/(2K) Σ_{i,j} ξ_{k,i}ξ_{k,j} ρ(s_i)ρ(s_j) on dynamic block (Eq. 7)."""
    ps = rho(s)
    xi_dyn = xi_k[n_input:]
    bilinear = float(np.sum(xi_dyn[:, None] * xi_dyn[None, :] * ps[:, None] * ps[None, :]))
    return -0.5 / max(rank, 1) * bilinear


def grad_xi_ki(
    s: NDArray[np.floating],
    xi_k: NDArray[np.floating],
    lambda_k: float,
    i: int,
    rank_k: int,
    n_input: int,
) -> float:
    """∂I/∂ξ_{k,i} (Eq. 8, digital evaluation)."""
    ps = rho(s)
    xi_dyn = xi_k[n_input:]
    ps_dyn = ps
    if i < n_input:
        ps_i = float(rho(np.array([0.0]))[0])  # clamped input channel
        inner = float(xi_dyn @ ps_dyn)
    else:
        ps_i = float(ps_dyn[i - n_input])
        inner = float(xi_dyn @ ps_dyn)
    return -lambda_k / max(rank_k, 1) * ps_i * inner
