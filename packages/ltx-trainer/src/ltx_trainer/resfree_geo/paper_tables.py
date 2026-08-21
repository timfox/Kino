"""Tables 2–9 excerpts from the paper."""

from __future__ import annotations

from typing import Any


def table2_beltrami_reconstruction() -> list[dict[str, Any]]:
    return [
        {"case": "(a) Eq. 23", "time_s": 0.183517, "mean_delta_d": 0.00943353, "mean_delta_mu": 0.01661039},
        {"case": "(b) Eq. 24", "time_s": 0.178444, "mean_delta_d": 0.00495606, "mean_delta_mu": 0.00628607},
        {"case": "(c) Eq. 25", "time_s": 0.181272, "mean_delta_d": 0.00333415, "mean_delta_mu": 0.00763160},
        {"case": "(d) Eq. 26", "time_s": 0.178142, "mean_delta_d": 0.00323314, "mean_delta_mu": 0.00575289},
    ]


def table3_resolution_beltrami() -> list[dict[str, Any]]:
    return [
        {"N": 48, "time_s": 0.224661, "mean_delta_d": 0.00191878, "mean_delta_mu": 0.02008131},
        {"N": 64, "time_s": 0.250878, "mean_delta_d": 0.00226898, "mean_delta_mu": 0.01994414},
        {"N": 80, "time_s": 0.246121, "mean_delta_d": 0.00282290, "mean_delta_mu": 0.02708278},
        {"N": 96, "time_s": 0.283674, "mean_delta_d": 0.00380885, "mean_delta_mu": 0.03982333},
    ]


def table4_deq_square_boundary() -> list[dict[str, Any]]:
    return [
        {"case": "(a) Eq. 28", "time_s": 0.804569, "std_rho_orig": 0.453812, "std_rho_map": 0.082541, "min_jacobian": 0.100071},
        {"case": "(b) Eq. 29", "time_s": 0.504526, "std_rho_orig": 0.397409, "std_rho_map": 0.064182, "min_jacobian": 0.085428},
        {"case": "(c) Eq. 30", "time_s": 0.649909, "std_rho_orig": 0.480728, "std_rho_map": 0.093411, "min_jacobian": 0.099442},
        {"case": "(d) Eq. 31", "time_s": 0.312899, "std_rho_orig": 0.349701, "std_rho_map": 0.048789, "min_jacobian": 0.238288},
    ]


def table5_deq_free_boundary() -> list[dict[str, Any]]:
    return [
        {"case": "(a) Eq. 28", "time_s": 0.251685, "std_rho_orig": 0.453812, "std_rho_map": 0.072977, "min_jacobian": 0.1117707},
        {"case": "(b) Eq. 29", "time_s": 0.210106, "std_rho_orig": 0.397409, "std_rho_map": 0.070202, "min_jacobian": 0.122339},
        {"case": "(c) Eq. 30", "time_s": 0.214805, "std_rho_orig": 0.480728, "std_rho_map": 0.087137, "min_jacobian": 0.121764},
        {"case": "(d) Eq. 31", "time_s": 0.206635, "std_rho_orig": 0.349701, "std_rho_map": 0.036166, "min_jacobian": 0.270623},
    ]


def table6_deq_resolution() -> list[dict[str, Any]]:
    return [
        {"N": 32, "time_s": 0.467233, "std_rho_orig": 0.391159, "std_rho_map": 0.090291, "min_jacobian": 0.155488},
        {"N": 64, "time_s": 0.367197, "std_rho_orig": 0.400335, "std_rho_map": 0.067485, "min_jacobian": 0.075382},
        {"N": 128, "time_s": 0.458393, "std_rho_orig": 0.402907, "std_rho_map": 0.078921, "min_jacobian": 0.073391},
        {"N": 256, "time_s": 0.407083, "std_rho_orig": 0.403503, "std_rho_map": 0.134581, "min_jacobian": 0.118330},
    ]


def table7_dem3d_cube_boundary() -> list[dict[str, Any]]:
    return [
        {"case": "(a) Eq. 33", "time_s": 0.758503, "std_rho_orig": 0.319948, "std_rho_map": 0.073687, "min_jacobian": 0.139083},
        {"case": "(b) Eq. 34", "time_s": 0.764084, "std_rho_orig": 0.327913, "std_rho_map": 0.063817, "min_jacobian": 0.236052},
        {"case": "(c) Eq. 36", "time_s": 0.755914, "std_rho_orig": 0.501315, "std_rho_map": 0.275038, "min_jacobian": 0.182880},
        {"case": "(d) Eq. 37", "time_s": 0.755548, "std_rho_orig": 0.366671, "std_rho_map": 0.110930, "min_jacobian": 0.019724},
    ]


def table8_dem3d_free_boundary() -> list[dict[str, Any]]:
    return [
        {"case": "(a) Eq. 33", "time_s": 0.991099, "std_rho_orig": 0.319948, "std_rho_map": 0.055559, "min_jacobian": 0.088221},
        {"case": "(b) Eq. 34", "time_s": 0.980939, "std_rho_orig": 0.327913, "std_rho_map": 0.079853, "min_jacobian": 0.337495},
        {"case": "(c) Eq. 36", "time_s": 0.952575, "std_rho_orig": 0.501315, "std_rho_map": 0.137543, "min_jacobian": 0.085793},
        {"case": "(d) Eq. 37", "time_s": 0.944139, "std_rho_orig": 0.366671, "std_rho_map": 0.100304, "min_jacobian": 0.050543},
    ]


def table9_dem3d_resolution() -> list[dict[str, Any]]:
    return [
        {"N": 32, "time_s": 0.411478, "std_rho_orig": 0.313550, "std_rho_map": 0.079741, "min_jacobian": 0.061818},
        {"N": 40, "time_s": 0.594614, "std_rho_orig": 0.316422, "std_rho_map": 0.073728, "min_jacobian": 0.018812},
        {"N": 56, "time_s": 1.315721, "std_rho_orig": 0.318893, "std_rho_map": 0.074921, "min_jacobian": 0.000006},
        {"N": 64, "time_s": 3.077169, "std_rho_orig": 0.319491, "std_rho_map": 0.075686, "min_jacobian": 0.000028},
    ]


def framework_comparison() -> list[dict[str, str]]:
    """Table 1 style overview."""
    return [
        {"framework": "FNO", "input_resolution": "Fixed uniform grid", "constraint": "Data-driven / PINN"},
        {"framework": "DeepONet", "input_resolution": "Fixed sensor locations", "constraint": "Data-driven / PINN"},
        {"framework": "RINO", "input_resolution": "Dictionary embedding", "constraint": "Data-driven mappings"},
        {"framework": "Proposed", "input_resolution": "Arbitrary, resolution-free", "constraint": "Variational energy"},
    ]
