"""SPIM Equilibrium Propagation configuration (Vanden Abeele et al. arXiv:2606.13454)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

PAPER_ARXIV = "2606.13454"
PAPER_TITLE = (
    "Optical Implementation of Equilibrium Propagation Using Spatial Photonic Ising Machines"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

WINE_TEST_ACC_EXP = 89.7
WINE_TEST_ACC_STD = 3.2
WINE_SIM_ACC = 98.2
MNIST_TEST_ACC = 97.81
MNIST_LAYERED_ACC = 97.71
FINITE_DIFF_DELTA = math.pi / 4


class PatternMode(str, Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass
class SPIMEPConfig:
    """Hybrid optical-digital EP on a rank-K Mattis SPIM."""

    n_input: int = 13
    n_hidden: int = 5
    n_output: int = 3
    rank: int = 20
    alpha: float = 2.0
    beta: float = 0.9
    n_free: int = 10
    n_nudge: int = 5
    inference_lr: float = 0.05
    learn_lr_lambda: float = 0.02
    learn_lr_xi: float = 0.0  # BOP in experiment; 0 disables digital xi updates in stub
    pattern_mode: PatternMode = PatternMode.BINARY
    bop_tau: float = 5e-8
    bop_gamma: float = 1e-4
    l2_lambda: float = 0.001
    finite_diff_delta: float = FINITE_DIFF_DELTA
    use_tilde_j: bool = True
    hybrid_digital_input: bool = False

    @property
    def n_dynamic(self) -> int:
        return self.n_hidden + self.n_output

    @property
    def n_total(self) -> int:
        return self.n_input + self.n_dynamic

    def n_spim_evaluations(self) -> int:
        """Eq. (11): SPIM shots per sample per training step."""
        nd = self.n_dynamic
        return 2 * nd * (self.n_free + 2 * self.n_nudge) + 1
