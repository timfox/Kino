"""Paper anchors — Branch-Level Energy Localization (Montoya et al., arXiv:2606.07076)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07076"
PAPER_TITLE = (
    "Branch-Level Energy Localization in Three-Phase Loads: "
    "Resolving Indeterminacy in Time-Domain"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "IEEE / arXiv preprint"
PAPER_CODE = None

F0_HZ = 50.0
OMEGA0 = 2.0 * 3.141592653589793 * F0_HZ

# Sec. IV test case registry
TEST_CASES: tuple[dict[str, str], ...] = (
    {"id": "IV-A", "name": "balanced_rl", "section": "IV-A"},
    {"id": "IV-B", "name": "resistive_open_phase", "section": "IV-B"},
    {"id": "IV-C", "name": "triac_switched_resistive", "section": "IV-C"},
    {"id": "IV-D", "name": "delta_wye_indeterminacy", "section": "IV-D"},
    {"id": "IV-E", "name": "fluctuating_phase", "section": "IV-E"},
    {"id": "IV-F", "name": "four_wire_nonlinear", "section": "IV-F"},
)

# Sec. IV-A balanced RL anchors (R=1.1 Ω, L=1 mH, V=120 V rms)
BALANCED_RL = {
    "V_rms": 120.0,
    "R_ohm": 1.1,
    "L_H": 0.001,
    "P_3phi_kW": 3.0 * 120.0 * 120.0 / 1.1 * 0.96,  # ~ P = 3VI cos θ order
    "phase_shift_2theta_deg": 32.0,
}

# Sec. IV-B de Leon–Cohen open-phase resistive
OPEN_PHASE_RAB = {
    "Rab_ohm": 0.65,
    "ic_zero": True,
}

# Sec. IV-C TRIAC switched delta (α = 60°)
TRIAC_LOAD = {
    "alpha_deg": 60.0,
    "Q_F_kvar_order": 14.6,
}

# Sec. IV-D topology indeterminacy
DELTA_WYE_INDETERMINACY = {
    "R_delta_ohm": (4.0, 6.0, 5.0),
    "L_delta_mH": (10.0, 12.0, 11.0),
    "R_wye_ohm": (0.56, 0.47, 0.59),
    "L_wye_mH": (2.22, 2.45, 2.63),
    "P_avg_kW": 98.7,
    "reconstruction_residual_order": 1e-16,
}

# Sec. IV-E fluctuating phase (Jeltsema lift)
FLUCTUATING_PHASE = {
    "V_rms": 1.0,
    "I_rms": 0.5,
    "omega1_over_omega0": 0.2,
    "gamma_rad": 1.5707963267948966,
    "p_alpha_beta_W": 0.708,
    "q_alpha_beta_mean": 0.0,
}

# Sec. IV-F four-wire nonlinear phase b duality
PHASE_B_DUALITY = {
    "Gb_S": 0.25,
    "Gamma_b_1_per_H": 100.0,
    "Cb_uF": 100.0,
    "energy_match_tol_W": 0.01,
}

THEOREMS = (
    "Theorem 1: Branch-Level Localization (unique decomposition given admissible topology T)",
    "Theorem 2: Topology Indeterminacy (distinct (T, θ) → same terminals, different branch maps)",
    "Theorem 3: Generalized Energetic Duality (series/parallel families share branch energy)",
)

TRAIN_DEFAULTS = {
    "balance": "p_term = p_R + W_L' + W_C' (Tellegen, Eq. 3)",
    "delta_model": "ixy = Gxy vxy + Γxy v̈xy + Cxy v'xy (Eq. 11)",
    "wye_model": "vxn = Rx ix + Lx i'x + Sx îx (Eq. 12)",
}
