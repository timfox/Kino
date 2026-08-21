"""MOC-inspired neural Q-factor controller (§2.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.biear.config import BiearConfig
from ltx_trainer.biear.filterbank import apply_q_control, delta_q_profile


def smooth_spl(
    spl: np.ndarray,
    *,
    beta: float,
    prev: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if prev is None:
        prev = spl.copy()
    smoothed = beta * prev + (1.0 - beta) * spl
    return smoothed, smoothed


def controller_delta(
    spl: np.ndarray,
    smoothed: np.ndarray,
    *,
    seed: int = 0,
    cfg: BiearConfig | None = None,
) -> np.ndarray:
    """Stub GRU+FC controller → δ[t,k] ∈ [-1,1]."""
    cfg = cfg or BiearConfig()
    rng = np.random.default_rng(seed)
    inp = np.concatenate([spl, smoothed])
    w = rng.normal(0, 0.02, size=(inp.shape[0], cfg.n_subbands))
    delta = np.tanh(inp @ w)
    return np.clip(delta, -1.0, 1.0)


def modulate_q_frame(
    q0: np.ndarray,
    spl_l: np.ndarray,
    spl_r: np.ndarray,
    *,
    dual: bool,
    relative: bool,
    seed: int = 0,
    cfg: BiearConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or BiearConfig()
    dq = delta_q_profile(cfg, relative=relative)
    sm_l, _ = smooth_spl(spl_l, beta=cfg.ema_beta)
    sm_r, _ = smooth_spl(spl_r, beta=cfg.ema_beta)
    if dual:
        d_l = controller_delta(spl_l, sm_l, seed=seed, cfg=cfg)
        d_r = controller_delta(spl_r, sm_r, seed=seed + 1, cfg=cfg)
    else:
        concat = np.concatenate([spl_l, spl_r])
        sm = np.concatenate([sm_l, sm_r])
        d_shared = controller_delta(concat, sm, seed=seed, cfg=cfg)
        d_l = d_r = d_shared
    return {
        "q_left": apply_q_control(q0, d_l, dq, relative=relative, cfg=cfg),
        "q_right": apply_q_control(q0, d_r, dq, relative=relative, cfg=cfg),
    }


def controller_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    rng = np.random.default_rng(seed)
    q0 = np.linspace(1.0, 8.0, cfg.n_subbands)
    spl_l = rng.uniform(0.01, 1.0, cfg.n_subbands)
    spl_r = rng.uniform(0.01, 1.0, cfg.n_subbands)
    out = modulate_q_frame(q0, spl_l, spl_r, dual=True, relative=True, seed=seed, cfg=cfg)
    return {
        "dual_controller": True,
        "relative_control": True,
        "ema_beta": cfg.ema_beta,
        "q_left_mean": float(np.mean(out["q_left"])),
        "q_right_mean": float(np.mean(out["q_right"])),
        "asymmetric_ears": float(np.mean(np.abs(out["q_left"] - out["q_right"]))) > 0.01,
    }
