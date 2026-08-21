"""Six Section-IV test cases (synthetic, 50 Hz)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.branch_energy.branch import (
    branch_balance_residual,
    electric_storage_rate,
    joule_dissipation_r,
    magnetic_storage_rate,
    terminal_power,
)
from ltx_trainer.branch_energy.config import BranchEnergyConfig
from ltx_trainer.branch_energy.constants import (
    BALANCED_RL,
    DELTA_WYE_INDETERMINACY,
    FLUCTUATING_PHASE,
    OPEN_PHASE_RAB,
    PHASE_B_DUALITY,
    TRIAC_LOAD,
)
from ltx_trainer.branch_energy.classical import (
    open_phase_classical_comparison,
    triac_classical_comparison,
)
from ltx_trainer.branch_energy.duality import (
    duality_invariant_error,
    parallel_branch_energy,
    series_branch_energy,
    series_from_parallel,
)


def _time_grid(cfg: BranchEnergyConfig) -> tuple[np.ndarray, float]:
    dt = 1.0 / cfg.fs_hz
    t = np.arange(0.0, cfg.num_periods / cfg.f0_hz, dt)
    return t, dt


def case_balanced_rl(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-A: aggregated invariance hides per-phase exchange."""
    cfg = cfg or BranchEnergyConfig(num_periods=2.0)
    t, dt = _time_grid(cfg)
    w = 2.0 * math.pi * cfg.f0_hz
    r, l = BALANCED_RL["R_ohm"], BALANCED_RL["L_H"]
    v_rms = BALANCED_RL["V_rms"]
    z_mag = math.sqrt(r * r + (w * l) ** 2)
    theta = math.atan2(w * l, r)
    i_peak = math.sqrt(2) * v_rms / z_mag

    phases = (0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
    w_l_dot = []
    p_r = []
    for th in phases:
        v = math.sqrt(2) * v_rms * np.sin(w * t + th)
        i = math.sqrt(2) * (v_rms / z_mag) * np.sin(w * t + th - theta)
        p_r.append(joule_dissipation_r(i, r))
        w_l_dot.append(magnetic_storage_rate(i, l, dt))

    w_l_sum = sum(w_l_dot)
    p3 = sum(p_r)
    return {
        "case": "IV-A",
        "aggregated_p_R_std": float(np.std(p3)),
        "aggregated_W_L_dot_rms": float(np.sqrt(np.mean(w_l_sum**2))),
        "per_phase_W_L_dot_rms": [float(np.sqrt(np.mean(w**2))) for w in w_l_dot],
        "balance_residual": branch_balance_residual(p3, p3, w_l_sum, np.zeros_like(p3)),
        "phase_shift_2theta_deg": math.degrees(2 * theta),
    }


def case_resistive_open_phase(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-B: single Rab, ic ≡ 0, all power is Joule."""
    cfg = cfg or BranchEnergyConfig(num_periods=1.0)
    t, dt = _time_grid(cfg)
    w = 2.0 * math.pi * cfg.f0_hz
    v_ll = 400.0 / math.sqrt(3.0) * math.sqrt(2)
    rab = OPEN_PHASE_RAB["Rab_ohm"]

    v_ab = v_ll * np.sqrt(3) * np.sin(w * t)
    v_bc = v_ll * np.sqrt(3) * np.sin(w * t - 2.0 * math.pi / 3.0)
    v_ca = v_ll * np.sqrt(3) * np.sin(w * t + 2.0 * math.pi / 3.0)
    i_ab = v_ab / rab
    ia = i_ab
    ib = -i_ab
    ic = np.zeros_like(ia)

    # Single dissipative branch: p_term = v_ab i_ab = R i_ab² (de Leon–Cohen open-phase).
    p_term = v_ab * i_ab
    p_r = joule_dissipation_r(i_ab, rab)
    residual = branch_balance_residual(p_term, p_r, np.zeros_like(p_term), np.zeros_like(p_term))
    cpc_cmp = open_phase_classical_comparison(cfg_fs=cfg.fs_hz, periods=cfg.num_periods)

    return {
        "case": "IV-B",
        "Rab_ohm": rab,
        "ic_max_abs": float(np.max(np.abs(ic))),
        "W_L_dot_max": 0.0,
        "W_C_dot_max": 0.0,
        "balance_residual": residual,
        "p_term_mean_kW": float(np.mean(p_term) / 1000.0),
        "cpc_ghost_in_open_phase": cpc_cmp["cpc_ghost_in_open_phase"],
        "cpc_ic_act_max": cpc_cmp["cpc_ic_act_max"],
    }


def case_triac_switched(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-C: switched resistive load, no storage."""
    cfg = cfg or BranchEnergyConfig(num_periods=2.0)
    t, dt = _time_grid(cfg)
    w = 2.0 * math.pi * cfg.f0_hz
    alpha = math.radians(TRIAC_LOAD["alpha_deg"])
    g = 1.0 / 10.0
    v = 400.0 * np.sin(w * t)
    firing = (w * t % (2.0 * math.pi)) >= alpha
    firing &= (w * t % (2.0 * math.pi)) <= math.pi + alpha
    i = np.where(firing, g * v, 0.0)
    p_r = joule_dissipation_r(i, 1.0 / g)
    p_term = terminal_power(v, i)
    # pseudo reactive from v_alpha i_beta asymmetry proxy
    q_proxy = float(np.sqrt(np.mean((v * np.roll(i, len(i) // 4)) ** 2)))
    triac_cmp = triac_classical_comparison(cfg_fs=cfg.fs_hz, periods=cfg.num_periods)

    return {
        "case": "IV-C",
        "alpha_deg": TRIAC_LOAD["alpha_deg"],
        "W_L_dot_rms": 0.0,
        "storage_rms": 0.0,
        "q_proxy_nontrivial": q_proxy > 1.0,
        "akagi_q_rms": triac_cmp["q_rms"],
        "pseudo_reactive_nontrivial": triac_cmp["pseudo_reactive_nontrivial"],
        "balance_residual": branch_balance_residual(p_term, p_r, np.zeros_like(p_term), np.zeros_like(p_term)),
        "p_term_mean_kW": float(np.mean(p_term) / 1000.0),
    }


def _rl_branch_series(
    v_peak: float,
    phase: float,
    r: float,
    l: float,
    w: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Series RL branch with v = V sin(ωt+phase); returns i, p_R, W'_L, v*i."""
    z = math.sqrt(r * r + (w * l) ** 2)
    phi = math.atan2(w * l, r)
    i = (v_peak / z) * np.sin(w * t + phase - phi)
    di = (v_peak / z) * w * np.cos(w * t + phase - phi)
    v = v_peak * np.sin(w * t + phase)
    p_r = r * i * i
    w_l = l * i * di
    return i, p_r, w_l, v * i


def case_delta_wye_indeterminacy(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-D: same terminals, distinct branch localizations."""
    cfg = cfg or BranchEnergyConfig(num_periods=2.0)
    t, dt = _time_grid(cfg)
    w = 2.0 * math.pi * cfg.f0_hz
    v_ll = 400.0 * math.sqrt(2)

    r_d = DELTA_WYE_INDETERMINACY["R_delta_ohm"]
    l_d = [x * 1e-3 for x in DELTA_WYE_INDETERMINACY["L_delta_mH"]]

    i_ab, p_r_ab, w_l_ab, p_ab = _rl_branch_series(v_ll, 0.0, r_d[0], l_d[0], w, t)
    i_bc, p_r_bc, w_l_bc, p_bc = _rl_branch_series(v_ll, -2.0 * math.pi / 3.0, r_d[1], l_d[1], w, t)
    i_ca, p_r_ca, w_l_ca, p_ca = _rl_branch_series(v_ll, 2.0 * math.pi / 3.0, r_d[2], l_d[2], w, t)

    ia = i_ab - i_ca
    ib = i_bc - i_ab
    v_ab = v_ll * np.sin(w * t)
    v_bc = v_ll * np.sin(w * t - 2.0 * math.pi / 3.0)

    p_r_delta = [p_r_ab, p_r_bc, p_r_ca]
    w_l_delta = [w_l_ab, w_l_bc, w_l_ca]
    p_term = p_ab + p_bc + p_ca

    r_w = DELTA_WYE_INDETERMINACY["R_wye_ohm"]
    l_w = [x * 1e-3 for x in DELTA_WYE_INDETERMINACY["L_wye_mH"]]
    i_a, i_b = ia, ib
    i_c = -(ia + ib)
    p_r_wye = [
        joule_dissipation_r(i_a, r_w[0]),
        joule_dissipation_r(i_b, r_w[1]),
        joule_dissipation_r(i_c, r_w[2]),
    ]
    w_l_wye = [
        magnetic_storage_rate(i_a, l_w[0], dt),
        magnetic_storage_rate(i_b, l_w[1], dt),
        magnetic_storage_rate(i_c, l_w[2], dt),
    ]

    p_r_d_sum = sum(p_r_delta)
    w_l_d_sum = sum(w_l_delta)
    residual = branch_balance_residual(p_term, p_r_d_sum, w_l_d_sum, np.zeros_like(p_term))

    p_r_w_sum = sum(p_r_wye)
    p_avg_delta = float(np.mean(p_r_d_sum) / 1000.0)
    p_avg_wye = float(np.mean(p_r_w_sum) / 1000.0)

    return {
        "case": "IV-D",
        "P_avg_kW_delta": p_avg_delta,
        "P_avg_kW_wye": p_avg_wye,
        "delta_p_R_peak_kW": float(np.max(p_r_ab) / 1000.0),
        "wye_p_R_peak_kW": float(np.max(p_r_wye[0]) / 1000.0),
        "profiles_differ": float(np.max(np.abs(p_r_ab - p_r_wye[0]))) > 1.0,
        "balance_residual": residual,
        "anchor_P_avg_kW": DELTA_WYE_INDETERMINACY["P_avg_kW"],
    }


def case_fluctuating_phase(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-E: Q→0 but manifest storage."""
    cfg = cfg or BranchEnergyConfig(f0_hz=50.0, fs_hz=200_000.0, num_periods=2.0)
    t, dt = _time_grid(cfg)
    w0 = 2.0 * math.pi * cfg.f0_hz
    w1 = w0 / 5.0
    v_rms = FLUCTUATING_PHASE["V_rms"]
    i_rms = FLUCTUATING_PHASE["I_rms"]
    gamma = FLUCTUATING_PHASE["gamma_rad"]

    phi = -gamma * np.sin(w1 * t)
    v = math.sqrt(2) * v_rms * np.cos(w0 * t)
    i = math.sqrt(2) * i_rms * np.cos(w0 * t - phi)

    g = (i * v) / (v * v + 1e-12)
    l_est = 0.01
    w_l = magnetic_storage_rate(i, l_est, dt)
    p_term = terminal_power(v, i)
    p_r = joule_dissipation_r(i, np.mean(g))

    return {
        "case": "IV-E",
        "p_mean_W": float(np.mean(p_term)),
        "W_L_dot_rms": float(np.sqrt(np.mean(w_l**2))),
        "fundamental_q_proxy": float(np.abs(np.mean(v * np.sin(w0 * t)) * np.mean(i * np.cos(w0 * t)))),
        "anchor_p_alpha_beta_W": FLUCTUATING_PHASE["p_alpha_beta_W"],
    }


def case_four_wire_duality(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    """Sec. IV-F phase-b series–parallel energetic duality."""
    cfg = cfg or BranchEnergyConfig(num_periods=2.0, fs_hz=20_000.0)
    t, dt = _time_grid(cfg)
    w = 2.0 * math.pi * cfg.f0_hz
    v = 120.0 * math.sqrt(2) * np.sin(w * t) + 0.12 * math.sqrt(2) * np.sin(3 * w * t)

    g = PHASE_B_DUALITY["Gb_S"]
    gamma = PHASE_B_DUALITY["Gamma_b_1_per_H"]
    c = PHASE_B_DUALITY["Cb_uF"] * 1e-6
    v_dot = np.gradient(v, dt)
    v_int = np.cumsum(v, dtype=np.float64) * dt
    i = g * v + gamma * v_int + c * v_dot

    par = parallel_branch_energy(v, i, g=g, gamma=gamma, c=c, dt=dt)
    rs, ls, ss = series_from_parallel(v, i, g_p=g, gamma_p=gamma, c_p=c, dt=dt)
    ser = series_branch_energy(v, i, rs, ls, ss, dt)
    err = duality_invariant_error(par, ser)

    return {
        "case": "IV-F",
        "duality_max_error_W": err,
        "anchor_tol_W": PHASE_B_DUALITY["energy_match_tol_W"],
        "balance_residual": branch_balance_residual(
            terminal_power(v, i), par["p_R"], par["W_L_dot"], par["W_C_dot"]
        ),
    }


def run_all_cases(cfg: BranchEnergyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BranchEnergyConfig()
    return {
        "IV-A": case_balanced_rl(cfg),
        "IV-B": case_resistive_open_phase(cfg),
        "IV-C": case_triac_switched(cfg),
        "IV-D": case_delta_wye_indeterminacy(cfg),
        "IV-E": case_fluctuating_phase(cfg),
        "IV-F": case_four_wire_duality(cfg),
    }
