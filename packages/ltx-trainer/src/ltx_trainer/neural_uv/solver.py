"""Neural chart solver stub (Algorithm 1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.neural_uv.config import NeuralUVConfig
from ltx_trainer.neural_uv.jacobian import chart_objective
from ltx_trainer.neural_uv.lbo import build_vertex_features
from ltx_trainer.neural_uv.siren import init_siren_weights, siren_forward
from ltx_trainer.neural_uv.tutte import tutte_embedding


def _flatten_params(params: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.concatenate([W.ravel() for W, _ in params] + [b.ravel() for _, b in params])


def _unflatten_params(flat: np.ndarray, template: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    off = 0
    for W, b in template:
        wn = W.size
        bn = b.size
        Wn = flat[off : off + wn].reshape(W.shape)
        off += wn
        bn_arr = flat[off : off + bn].reshape(b.shape)
        off += bn
        out.append((Wn, bn_arr))
    return out


def barrier_schedule(step: int, total: int, cfg: NeuralUVConfig) -> float:
    """Exponential anneal α(s) from 10 → 0.1."""
    if total <= 1:
        return cfg.barrier_alpha_end
    t = step / (total - 1)
    log_start = np.log(cfg.barrier_alpha_start)
    log_end = np.log(cfg.barrier_alpha_end)
    return float(np.exp(log_start + t * (log_end - log_start)))


def solve_chart_stub(
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: NeuralUVConfig | None = None,
    *,
    seed: int = 0,
    iters: int | None = None,
) -> dict[str, Any]:
    """
    CPU stub of Algorithm 1: Tutte warm-up → finite-difference Adam-like steps
    with flip rejection (restore on invalid step).
    """
    cfg = cfg or NeuralUVConfig()
    rng = np.random.default_rng(seed)
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    feats = build_vertex_features(verts, faces, k=cfg.spectral_rank)
    params = init_siren_weights(
        feats.shape[1],
        layers=cfg.siren_layers,
        width=cfg.siren_width,
        omega0=cfg.omega0,
        seed=seed,
    )
    # Tutte warm-up target
    uv_tutte = tutte_embedding(verts, faces)
    flat = _flatten_params(params)
    # Pretrain head toward Tutte (linear least squares on last layer only — stub)
    uv_init = siren_forward(feats, params, omega0=cfg.omega0)
    residual = uv_tutte - uv_init
    W_out, b_out = params[-1]
    params[-1] = (W_out, b_out + residual.mean(axis=0))
    flat = _flatten_params(params)
    n_iters = iters if iters is not None else min(cfg.adam_iters, 200)
    lr = 5e-4
    best_flat = flat.copy()
    best_metrics = chart_objective(siren_forward(feats, params, omega0=cfg.omega0), verts, faces, alpha=0.0)
    rejected = 0
    for step in range(n_iters):
        alpha = barrier_schedule(step, n_iters, cfg)
        eps = 1e-3 * (0.5 ** (step // 50))
        grad = np.zeros_like(flat)
        base_params = _unflatten_params(flat, params)
        base_uv = siren_forward(feats, base_params, omega0=cfg.omega0)
        base_loss = chart_objective(base_uv, verts, faces, alpha=alpha)["total"]
        for i in range(min(len(flat), 64)):
            trial = flat.copy()
            trial[i] += eps
            trial_params = _unflatten_params(trial, params)
            trial_uv = siren_forward(feats, trial_params, omega0=cfg.omega0)
            trial_loss = chart_objective(trial_uv, verts, faces, alpha=alpha)["total"]
            grad[i] = (trial_loss - base_loss) / eps
        flat_new = flat - lr * grad
        new_params = _unflatten_params(flat_new, params)
        new_uv = siren_forward(feats, new_params, omega0=cfg.omega0)
        metrics = chart_objective(new_uv, verts, faces, alpha=alpha)
        if metrics["flip_pct"] > 0.0:
            rejected += 1
            lr *= 0.5
            continue
        flat = flat_new
        if metrics["total"] < best_metrics["total"]:
            best_flat = flat.copy()
            best_metrics = metrics
    final_params = _unflatten_params(best_flat, params)
    final_uv = siren_forward(feats, final_params, omega0=cfg.omega0)
    final_metrics = chart_objective(final_uv, verts, faces, alpha=cfg.barrier_alpha_end)
    if final_metrics["flip_pct"] > 0.0:
        final_uv = uv_tutte
        final_metrics = chart_objective(final_uv, verts, faces, alpha=cfg.barrier_alpha_end)
    return {
        "uv": final_uv,
        "metrics": final_metrics,
        "rejected_steps": rejected,
        "iters": n_iters,
        "params_flat": best_flat,
    }


def solver_demo(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.stack([np.cos(angles), np.sin(angles), rng.normal(0, 0.03, n)], axis=1)
    faces = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int64)
    cfg = NeuralUVConfig(adam_iters=80, spectral_rank=8, siren_width=32, siren_layers=3)
    out = solve_chart_stub(verts, faces, cfg, seed=seed, iters=80)
    return {
        "flip_pct": out["metrics"]["flip_pct"],
        "esd": out["metrics"]["esd"],
        "min_det": out["metrics"]["min_det"],
        "rejected_steps": out["rejected_steps"],
    }
