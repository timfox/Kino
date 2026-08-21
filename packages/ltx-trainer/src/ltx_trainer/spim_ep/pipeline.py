"""Training and evaluation pipeline for hybrid SPIM-EP."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.spim_ep.config import SPIMEPConfig
from ltx_trainer.spim_ep.datasets import load_wine
from ltx_trainer.spim_ep.equilibrium import ep_parameter_update, predict_class, relax_to_equilibrium
from ltx_trainer.spim_ep.mattis import init_mattis
from ltx_trainer.spim_ep.metrics import accuracy


def train_one_sample(
    u: np.ndarray,
    y: np.ndarray,
    params,
    cfg: SPIMEPConfig,
    s_state: np.ndarray,
    xi_grad_ema: np.ndarray | None,
) -> tuple[Any, np.ndarray, np.ndarray | None, dict[str, float]]:
    """Single EP training step: free → nudged± → parameter update."""
    rng = np.random.default_rng()
    s0 = s_state + 0.01 * rng.normal(size=s_state.shape)
    s_free, n_free = relax_to_equilibrium(s0, u, None, params, cfg, beta=0.0, n_steps=cfg.n_free)
    s_plus, n_plus = relax_to_equilibrium(s_free, u, y, params, cfg, beta=cfg.beta, n_steps=cfg.n_nudge)
    s_minus, n_minus = relax_to_equilibrium(s_free, u, y, params, cfg, beta=-cfg.beta, n_steps=cfg.n_nudge)
    new_params, xi_grad_ema, upd = ep_parameter_update(
        params, s_free, s_plus, s_minus, cfg, xi_grad_ema=xi_grad_ema
    )
    n_spim = n_free + n_plus + n_minus + 1
    stats = {
        **upd,
        "n_spim_evaluations": float(n_spim),
        "expected_n_spim": float(cfg.n_spim_evaluations()),
    }
    return new_params, s_free, xi_grad_ema, stats


def evaluate_wine(
    cfg: SPIMEPConfig,
    *,
    seed: int = 0,
    train_epochs: int = 2,
    max_train: int = 40,
    max_test: int = 36,
) -> dict[str, Any]:
    """CPU stub: short Wine EP training + test accuracy."""
    data = load_wine(seed=seed)
    cfg = SPIMEPConfig(
        n_input=data["n_features"],
        n_hidden=cfg.n_hidden,
        n_output=data["n_classes"],
        rank=cfg.rank,
        alpha=cfg.alpha,
        beta=cfg.beta,
        n_free=min(cfg.n_free, 5),
        n_nudge=min(cfg.n_nudge, 3),
        pattern_mode=cfg.pattern_mode,
    )
    params, _ = init_mattis(cfg, seed=seed)
    nd = cfg.n_dynamic
    s_state = np.zeros(nd, dtype=np.float64)
    xi_ema: np.ndarray | None = None
    x_train = data["x_train"][:max_train]
    y_train = data["y_train"][:max_train]
    y_train_idx = data["y_train_idx"][:max_train]
    total_spim = 0.0
    for _ in range(train_epochs):
        for i in range(x_train.shape[0]):
            params, s_state, xi_ema, st = train_one_sample(
                x_train[i], y_train[i], params, cfg, s_state, xi_ema
            )
            total_spim += st["n_spim_evaluations"]
    preds = []
    for i in range(min(max_test, data["x_test"].shape[0])):
        s, _ = relax_to_equilibrium(
            s_state.copy(),
            data["x_test"][i],
            None,
            params,
            cfg,
            beta=0.0,
            n_steps=cfg.n_free,
        )
        preds.append(predict_class(s, cfg.n_output))
    y_test_idx = data["y_test_idx"][: len(preds)]
    acc = accuracy(y_test_idx, np.array(preds, dtype=int))
    return {
        "config": {
            "n_input": cfg.n_input,
            "n_dynamic": cfg.n_dynamic,
            "rank": cfg.rank,
            "n_spim_per_step": cfg.n_spim_evaluations(),
        },
        "data_source": data["source"],
        "train_samples": int(x_train.shape[0]),
        "test_accuracy": acc,
        "total_spim_evaluations": total_spim,
        "ref_wine_test_acc_pct": 89.7,
    }


def evaluation_demo_run(cfg: SPIMEPConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or SPIMEPConfig()
    return evaluate_wine(cfg, seed=seed, train_epochs=1, max_train=12, max_test=12)
