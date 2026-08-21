"""Double preconditioning step: AP then GP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ltx_trainer.dopr.config import DoPrConfig, GpKind
from ltx_trainer.dopr.gp import gradient_precondition
from ltx_trainer.dopr.preconditioning import activation_precondition


@dataclass
class DoPrState:
    step: int = 0
    gp_state: dict = field(default_factory=dict)
    grad_ema: np.ndarray | None = None
    cov_ema: np.ndarray | None = None


def dopr_step(
    weight: np.ndarray,
    gradient: np.ndarray,
    activations: np.ndarray,
    *,
    cfg: DoPrConfig | None = None,
    state: DoPrState | None = None,
) -> tuple[np.ndarray, DoPrState]:
    cfg = cfg or DoPrConfig()
    state = state or DoPrState()

    g = gradient
    if cfg.use_grad_ema and state.grad_ema is not None:
        g = cfg.ema_beta * state.grad_ema + (1.0 - cfg.ema_beta) * gradient
    state.grad_ema = g.copy()

    sigma = (activations.T @ activations) / float(activations.shape[0])
    if cfg.use_cov_ema and state.cov_ema is not None:
        sigma = cfg.ema_beta * state.cov_ema + (1.0 - cfg.ema_beta) * sigma
    state.cov_ema = sigma.copy()

    m = activation_precondition(
        g,
        activations,
        damping=cfg.damping,
        damping_mode=cfg.damping_mode,
    )
    d, state.gp_state = gradient_precondition(
        m,
        cfg.gp,
        state=state.gp_state,
        step=state.step + 1,
    )
    state.step += 1
    wd = 1.0 - cfg.learning_rate * cfg.weight_decay
    return weight * wd - cfg.learning_rate * d, state


def baseline_gp_step(
    weight: np.ndarray,
    gradient: np.ndarray,
    *,
    gp: GpKind = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    state: DoPrState | None = None,
) -> tuple[np.ndarray, DoPrState]:
    state = state or DoPrState()
    d, state.gp_state = gradient_precondition(
        gradient,
        gp,
        state=state.gp_state,
        step=state.step + 1,
    )
    state.step += 1
    wd = 1.0 - learning_rate * weight_decay
    return weight * wd - learning_rate * d, state
