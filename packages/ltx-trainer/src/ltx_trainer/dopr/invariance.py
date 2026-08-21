"""Affine invariance of AP / DoPr (Proposition 4.2, Figure 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.dopr.config import DoPrConfig
from ltx_trainer.dopr.optimizer import DoPrState, baseline_gp_step, dopr_step


@dataclass(frozen=True)
class InvarianceRun:
    losses_original: list[float]
    losses_transformed: list[float]
    max_abs_diff: float


def _mlp_forward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    h = np.maximum(0.0, x @ w1.T)
    return h @ w2.T


def _mlp_loss(x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> float:
    pred = _mlp_forward(x, w1, w2)
    return float(np.mean((pred - y) ** 2))


def _grad_w1(x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    h_pre = x @ w1.T
    h = np.maximum(0.0, h_pre)
    pred = h @ w2.T
    err = pred - y
    grad_h = err @ w2
    grad_h[h_pre <= 0.0] = 0.0
    return grad_h.T @ x / x.shape[0]


def run_affine_invariance_demo(
    *,
    steps: int = 30,
    batch: int = 128,
    seed: int = 0,
    use_dopr: bool = True,
) -> InvarianceRun:
    rng = np.random.RandomState(seed)
    d_in, h, d_out = 8, 16, 4
    x = rng.randn(batch, d_in)
    y = rng.randn(batch, d_out)

    a = rng.randn(d_in, d_in)
    q, _ = np.linalg.qr(a)
    d_vec = rng.randn(d_in)
    x_t = x @ q.T + d_vec

    w1 = rng.randn(h, d_in) * 0.1
    w2 = rng.randn(d_out, h) * 0.1
    w1_t = w1 @ np.linalg.inv(q)
    b1 = -w1 @ np.linalg.inv(q) @ d_vec
    w1_t = w1_t  # bias-free stub; shift absorbed in x_t centering

    cfg = DoPrConfig(gp="sgd", learning_rate=0.05, damping=1e-6)
    state_o = DoPrState()
    state_t = DoPrState()

    losses_o: list[float] = []
    losses_t: list[float] = []

    for _ in range(steps):
        g1 = _grad_w1(x, y, w1, w2)
        g1_t = _grad_w1(x_t, y, w1_t, w2)
        if use_dopr:
            w1, state_o = dopr_step(w1, g1, x, cfg=cfg, state=state_o)
            w1_t, state_t = dopr_step(w1_t, g1_t, x_t, cfg=cfg, state=state_t)
        else:
            w1, state_o = baseline_gp_step(w1, g1, gp="sgd", learning_rate=0.05, state=state_o)
            w1_t, state_t = baseline_gp_step(w1_t, g1_t, gp="sgd", learning_rate=0.05, state=state_t)
        losses_o.append(_mlp_loss(x, y, w1, w2))
        losses_t.append(_mlp_loss(x_t, y, w1_t, w2))

    diffs = [abs(a - b) for a, b in zip(losses_o, losses_t, strict=True)]
    return InvarianceRun(losses_original=losses_o, losses_transformed=losses_t, max_abs_diff=max(diffs))
