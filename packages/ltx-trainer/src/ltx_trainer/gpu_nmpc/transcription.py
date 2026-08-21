"""Direct transcription + InfiniteSIMD-NLP pattern accounting (Sec. 2–3.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TranscribedOcp:
    """Compact parametric NLP ˆP(p) from Eq. (6)."""

    n_primal: int
    n_eq: int
    n_ineq: int
    n_params: int
    horizon_nodes: int
    unique_patterns: int

    def kkt_dim(self) -> int:
        return self.n_primal + 2 * self.n_ineq + self.n_eq


def transcribe_scalar_ocp(*, horizon_nodes: int, n_state: int = 1, n_control: int = 1) -> TranscribedOcp:
    """Tutorial-style OCP from Code Snippet 1 (scalar state + control)."""
    n_y = horizon_nodes * (n_state + n_control)
    n_eq = horizon_nodes  # derivative defects + initial condition folded
    n_ineq = horizon_nodes * 2  # control bounds
    return TranscribedOcp(
        n_primal=n_y,
        n_eq=n_eq,
        n_ineq=n_ineq,
        n_params=2,  # y0, setpoint phase
        horizon_nodes=horizon_nodes,
        unique_patterns=3,
    )


def transcribe_distillation(*, n_trays: int, horizon_nodes: int) -> TranscribedOcp:
    """Problem (29) size accounting (structure only)."""
    n_state = 2 * n_trays  # x and y trays
    n_control = 4  # u, L, V, S
    n_y = horizon_nodes * (n_state + n_control)
    n_eq = horizon_nodes * (n_trays + 1)
    n_ineq = horizon_nodes * (2 * n_trays + n_control + 3)
    return TranscribedOcp(
        n_primal=n_y,
        n_eq=n_eq,
        n_ineq=n_ineq,
        n_params=n_trays + 4,
        horizon_nodes=horizon_nodes,
        unique_patterns=7,
    )


def build_condensed_spd(
    ocp: TranscribedOcp,
    *,
    seed: int,
    reg: float,
) -> np.ndarray:
    """Synthetic PD condensed Lifted-KKT matrix K^l_c — Eq. (17)."""
    rng = np.random.default_rng(seed)
    n = ocp.n_primal
    a = rng.standard_normal((n, n)) * 0.05
    k = a @ a.T + reg * np.eye(n)
    return 0.5 * (k + k.T)
