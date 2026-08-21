"""Paper anchors — Latent Reasoning Guidance (Bitan et al., arXiv:2606.05518)."""

from __future__ import annotations

PAPER_ARXIV = "2606.05518"
PAPER_TITLE = "Latent Reasoning Guidance for Parallel Code Translation"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
UPSTREAM_REPO = "https://github.com/Scientific-Computing-Lab/Parallax.git"

PRIMARY_MODEL = "LLaMA-3.3-70B"
PRM_BACKBONE = "Qwen-Coder-7B"
GENERATOR_HIDDEN_DIM = 8192
PRM_HIDDEN_DIM = 3584

LATENT_STEPS_TRAIN = 6
LATENT_STEPS_INFER = 12
BRANCHES_TRAIN = 3
BRANCHES_TEST = 8
PERTURBATION_STD = 1.42

PARATRANS_TEST_TASKS = 76
PARATRANS_DIRECTIONS: tuple[dict[str, str | int], ...] = (
    {"abbr": "C→O", "from_api": "CUDA", "to_api": "OpenMP", "count": 18},
    {"abbr": "O→C", "from_api": "OpenMP", "to_api": "CUDA", "count": 19},
    {"abbr": "S→O", "from_api": "Serial", "to_api": "OpenMP", "count": 20},
    {"abbr": "S→C", "from_api": "Serial", "to_api": "CUDA", "count": 19},
)

REPAIR_ATTEMPTS = 3
RUNS_MAIN = 3

# Table 1 — three-run mean validation rate (%).
TABLE1_VALIDATION: tuple[dict[str, str | float | None], ...] = (
    {"method": "Vanilla", "no_repair": 14.55, "no_repair_std": 2.55, "repair": 30.67, "repair_std": 4.34},
    {"method": "Fine-tuned", "no_repair": 33.48, "no_repair_std": 1.11, "repair": 36.00, "repair_std": 3.11},
    {"method": "Latent reasoning", "no_repair": 32.89, "no_repair_std": 3.44, "repair": 36.40, "repair_std": 5.47},
    {"method": "Random branch selection", "no_repair": 27.28, "no_repair_std": 6.34, "repair": None, "repair_std": None},
    {"method": "Latent PRM guidance", "no_repair": 42.10, "no_repair_std": 2.28, "repair": 45.18, "repair_std": 3.31},
)

# Paired latent PRM vs unguided latent reasoning.
PRM_VS_LATENT_GAIN_PP = {"no_repair": 9.21, "repair": 8.78}
PRM_VS_LATENT_CI = {"no_repair": (3.07, 15.79), "repair": (2.63, 15.35)}
PRM_VS_LATENT_P = {"no_repair": 0.0081, "repair": 0.0064}

# Table 2 — held-out branch selection (500 instances).
TABLE2_BRANCH_SELECTION: tuple[dict[str, str | float | None], ...] = (
    {"selector": "Original", "best_pct": 33.66, "gt_orig_pct": None, "ge_orig_pct": None},
    {"selector": "Random", "best_pct": 32.14, "best_std": 0.46, "gt_orig_pct": 43.0, "ge_orig_pct": 66.9},
    {"selector": "Latent PRM", "best_pct": 36.25, "best_std": 0.41, "gt_orig_pct": 45.0, "ge_orig_pct": 79.2},
)

# Table 4 — task-level win/loss/tie vs unguided latent reasoning.
TABLE4_WLT = {"no_repair": {"wins": 20, "ties": 50, "losses": 6}, "repair": {"wins": 18, "ties": 53, "losses": 5}}

# Table 5 — direction-wise compilation/validation (LR vs LR-PRM).
TABLE5_DIRECTION: tuple[dict[str, str | float], ...] = (
    {"direction": "C→O", "lr_compile": 54.4, "lr_valid": 40.4, "prm_compile": 52.6, "prm_valid": 43.9},
    {"direction": "O→C", "lr_compile": 40.4, "lr_valid": 33.3, "prm_compile": 47.4, "prm_valid": 36.8},
    {"direction": "S→O", "lr_compile": 63.3, "lr_valid": 33.3, "prm_compile": 65.0, "prm_valid": 55.0},
    {"direction": "S→C", "lr_compile": 40.4, "lr_valid": 24.6, "prm_compile": 42.1, "prm_valid": 24.6},
)

# Table 6 — latent search budget ablation.
TABLE6_ABLATION: tuple[dict[str, int | float], ...] = (
    {"b_test": 3, "k": 6, "validation_pct": 28.94},
    {"b_test": 3, "k": 12, "validation_pct": 35.52},
    {"b_test": 8, "k": 12, "validation_pct": 42.10},
)

# Trajectory analysis (952-sample PRM training set).
TRAJECTORY_STATS = {
    "samples_with_better_branch_pct": 95.0,
    "candidate_improves_unperturbed_pct": 59.0,
    "branch_only_success_pct": 25.0,
}

REWARD_WEIGHTS = {"compile": 0.30, "execute": 0.25, "validators": 0.45}
