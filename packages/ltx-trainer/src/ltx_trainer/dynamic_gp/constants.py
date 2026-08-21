"""Paper anchors — Dynamic Gaussian Processes (van Hulst et al., arXiv:2606.06705)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06705"
PAPER_TITLE = "Estimating Evolving Functions with Dynamic Gaussian Processes"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "SIAM journal preprint"
PAPER_CODE = "https://github.com/JvHulst/Dynamic-Gaussian-Processes"

# Sec. 6.1 heat equation — error vs M (Fig. 6.2 qualitative)
HEAT_ERROR_BY_M: tuple[tuple[int, float], ...] = (
    (3, 2.8),
    (9, 1.4),
    (31, 0.55),
    (101, 0.35),
)

# Sec. 6.2 wave equation
WAVE_EXAMPLE = {
    "domain": "[-10, 10]",
    "wave_speed_c": 2.0,
    "dt": 0.2,
    "M_fourier": 31,
    "D_state": 2,
    "p_observations": 3,
    "sigma_w_sq": 1e-5,
}

ERROR_DECOMPOSITION = (
    "noise_limited: tr((I_D ⊗ Λ_U) Ψ∞)",
    "leakage_gap: tr((I_D ⊗ Λ_U)(P∞ − Ψ∞))",
    "out_of_subspace: E[∥π⊥_M f_t∥²_L2]",
)

TRAIN_DEFAULTS = {
    "basis": "Fourier (orthonormal on bounded domain)",
    "reduction": "separable kernel → finite-dimensional Kalman filter on z_t ∈ R^{DM}",
    "update": "DGP Eq. (3.7)",
    "prediction": "DGP Eq. (3.8)",
}
