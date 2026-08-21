"""NPC framework card, channel examples, and rate–blocklength smoke (arXiv:2605.25699)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.npc.bounds import (
    epsilon_capacity,
    gaussian_approx_log_m_3x4,
    gaussian_approx_log_m_bsc,
    log_m_star_achievability_lower,
    log_m_star_converse_upper,
    normalized_rate,
)
from ltx_trainer.npc.config import NPCConfig
from ltx_trainer.npc.gaussian import (
    channel_3x4_variance,
    lattice_gaussian_average,
    solve_bsc_c_epsilon,
    solve_c_epsilon,
)
from ltx_trainer.npc.geometry import (
    affine_output_dimension,
    bsc_relative_volume_ratio,
    channel_3x4_affine_map,
    channel_3x4_example,
    channel_3x4_relative_volume_ratio,
    num_transfer_directions,
    pmin,
)
from ltx_trainer.npc.lattice import (
    enumerate_simplex_lattice,
    filter_kw_3x4,
    monte_carlo_decode_error,
)


def framework_card(cfg: NPCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NPCConfig()
    W = channel_3x4_example()
    d = affine_output_dimension(W)
    return {
        "name": "Noisy permutation channel (finite blocklength)",
        "paper": "arXiv:2605.25699",
        "venue": "IEEE Trans. Commun. (submitted)",
        "affine_dimension": d,
        "transfer_directions": num_transfer_directions(d),
        "pmin": pmin(W),
        "lambda_star_3x4": channel_3x4_relative_volume_ratio(),
        "epsilon_capacity": epsilon_capacity(d),
        "achievability": "simplex lattice on aff(P_W) + Euclidean NN decoder",
        "converse": "KL covering + modified meta-converse + local testing",
        "config": {
            "epsilon": cfg.epsilon,
            "covering_a": cfg.covering_a,
        },
    }


def bsc_rate_points(
    delta: float = 0.11,
    epsilon: float = 1e-3,
    *,
    n_values: list[int] | None = None,
) -> dict[str, Any]:
    """Sec. VI-A: normalized rates for BSC (Fig. 1 style)."""
    n_values = n_values or [200, 400, 800, 1200, 1600, 2000]
    c_delta = solve_bsc_c_epsilon(delta, epsilon)
    lam = bsc_relative_volume_ratio(delta)
    rows = []
    for n in n_values:
        log_ach = gaussian_approx_log_m_bsc(n, c_delta)
        log_conv = log_m_star_converse_upper(n, 1, lam, a=1.0 / c_delta)
        rows.append(
            {
                "n": n,
                "log_m_ach_approx": log_ach,
                "log_m_conv": log_conv,
                "rate_ach": normalized_rate(log_ach, n),
                "rate_conv": normalized_rate(log_conv, n),
            }
        )
    return {
        "delta": delta,
        "epsilon": epsilon,
        "c_delta_epsilon": c_delta,
        "lambda_star": lam,
        "capacity": epsilon_capacity(1),
        "points": rows,
    }


def channel_3x4_rate_points(
    epsilon: float = 1e-2,
    *,
    n_values: list[int] | None = None,
) -> dict[str, Any]:
    """Sec. VI-B: 3×4 lower-dimensional channel (Fig. 2 style)."""
    n_values = n_values or [200, 400, 600, 800, 1000, 1200]
    d = 2
    lam = channel_3x4_relative_volume_ratio()
    lattice = filter_kw_3x4(enumerate_simplex_lattice(d, 8))
    pts = [u for u in lattice]
    c_eps = solve_c_epsilon(
        epsilon,
        channel_3x4_variance,
        pts,
        d,
    )
    rows = []
    for n in n_values:
        log_ref = gaussian_approx_log_m_3x4(n, c_eps, lam, d=d)
        log_ach_lead = log_m_star_achievability_lower(n, d, c_eps, lam)
        log_conv = log_m_star_converse_upper(n, d, lam, a=1.0 / c_eps)
        N_try = max(1, int(c_eps * (n**0.5)))
        alat = lattice_gaussian_average(
            n,
            N_try,
            pts,
            channel_3x4_variance,
            d,
        )
        rows.append(
            {
                "n": n,
                "N_guess": N_try,
                "A_lat": alat,
                "log_m_ref_approx": log_ref,
                "log_m_ach_leading": log_ach_lead,
                "log_m_conv": log_conv,
                "rate_ref": normalized_rate(log_ref, n),
                "rate_ach": normalized_rate(log_ach_lead, n),
                "rate_conv": normalized_rate(log_conv, n),
            }
        )
    return {
        "epsilon": epsilon,
        "c_epsilon": c_eps,
        "lambda_star": lam,
        "d": d,
        "lattice_points_kw": len(pts),
        "capacity": epsilon_capacity(d),
        "points": rows,
    }


def evaluation_demo(*, trials: int = 50, n: int = 80, N: int = 6) -> dict[str, Any]:
    """Small Monte Carlo decode smoke for Sec. VI-B."""
    d = 2
    lattice = filter_kw_3x4(enumerate_simplex_lattice(d, N))
    laws = [channel_3x4_affine_map(u) for u in lattice]
    pe = monte_carlo_decode_error(n, N, lattice, laws, trials=trials)
    return {
        "n": n,
        "N": N,
        "messages": lattice.shape[0],
        "monte_carlo_Pe": pe,
        "trials_per_message": trials,
    }


def table_main_results() -> dict[str, Any]:
    """Leading terms matching achievability / converse (Sec. V–VI)."""
    W = channel_3x4_example()
    d = affine_output_dimension(W)
    return {
        "achievability_leading": f"d log(c√n) + log λ* - log d!,  d={d}",
        "converse_leading": f"d log(√n/a) + log λ* + O(1)",
        "C_epsilon": epsilon_capacity(d),
        "BSC_d": 1,
        "BSC_lambda_star": "1 - 2δ",
    }
