"""Paper anchors — HD LoRA fine-tuning in solvable attention (Duranthon et al., arXiv:2606.05899)."""

from __future__ import annotations

PAPER_ARXIV = "2606.05899"
PAPER_TITLE = "High-Dimensional Theory of LoRA Fine-Tuning in a Solvable Attention Model"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "EPFL / arXiv preprint"
PAPER_CODE = None

# Asymptotic ratios Eq. (7)
ASYMPTOTIC = {
    "P0_over_D": "kappa0 = Theta(1)",
    "P_over_D": "kappa = Theta(1), kappa >= 1",
    "N_over_D2": "alpha = Theta(1)",
    "Nprime_over_D": "alpha_prime = Theta(1)",
    "T": "Theta(1) tokens",
}

# Fig. 1 anchor (independent sequences e=0)
FIG1 = {
    "alpha": 0.1,
    "alpha_prime": 3.0,
    "T": 3,
    "kappa0": 1.0,
    "kappa": 1.0,
    "delta": 0.5,
    "lambda_prime_grid": (0.05, 0.5, 1.0),
    "lambda_grid": (0.01, 0.1, 1.0),
    "D_sim": 150,
}

FIG2 = {
    "T": 3,
    "lambda_prime": 0.5,
    "alpha_prime_grid": (0.5, 1.0, 2.0, 5.0, 10.0),
}

FIG3 = {
    "T": 2,
    "delta": 0.5,
    "kappa0": 0.5,
    "kappa": 1.0,
    "alpha": 0.2,
    "alpha_prime": 3.0,
    "e": 1,
}

FIG4 = {
    "T": 3,
    "delta": 0.0,
    "alpha_bar_prime": 100.0,
    "rho_grid": (0.04, 0.02, 0.015),
}

RESULTS = (
    "Result 1: test pre-activations (z*, z) ~ Q, (chi*, chi) ~ Q'",
    "Result 2: train pre-activations via proximal Prox / prox",
    "Result 3: E, E', L, L' and overlap o_w = m/sqrt(q q0)",
    "Result 4: pre-training order parameters (Q, M, V) from Phi",
    "Result 5: fine-tuning order parameters (q, m, v) from phi",
    "Result 6: Bayes-optimal E_BO, E'_BO overlaps Q_BO, q_BO",
)

TRAIN_DEFAULTS = {
    "architecture": "tied K=Q, identity V, rank-one LoRA on frozen extensive-rank W",
    "pretrain_loss": "L(W) = sum D^{-1}||y_hat - y||_F^2 + lambda||W||_F^2",
    "finetune_loss": "L'(w) = sum D^{-1}||y_hat(W,w) - y'||_F^2 + lambda'||w||_2^2",
    "effective_noise": "Delta_eff = Delta/2 + Q0 - 2M + Q (linear sigma, e=0)",
}
