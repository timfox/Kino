"""Mass-spring LTI plant (Section IV-A)."""

from __future__ import annotations

import math

import numpy as np

from ltx_trainer.timesfm_attack.config import MassSpringConfig


def mass_spring_matrices(cfg: MassSpringConfig) -> tuple[np.ndarray, np.ndarray]:
    w = cfg.omega
    dt = cfg.dt
    a = np.array(
        [
            [math.cos(w * dt), math.sin(w * dt) / w],
            [-w * math.sin(w * dt), math.cos(w * dt)],
        ],
        dtype=np.float64,
    )
    c = np.array(
        [
            [1.0, 0.0],
            [0.31, -0.48],
            [-0.21, 0.43],
        ],
        dtype=np.float64,
    )
    return a, c


def simulate_mass_spring(cfg: MassSpringConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simulate x[k+1]=Ax[k], y[k]=Cx[k]+v[k] with x[0]=[1,0]."""
    a, c = mass_spring_matrices(cfg)
    rng = np.random.RandomState(cfg.seed)
    n = cfg.steps
    m = c.shape[0]
    x = np.zeros((n + 1, 2), dtype=np.float64)
    y = np.zeros((n + 1, m), dtype=np.float64)
    x[0] = np.array([1.0, 0.0])
    for k in range(n + 1):
        y[k] = c @ x[k] + rng.randn(m) * cfg.sigma_v
        if k < n:
            x[k + 1] = a @ x[k]
    return x, y
