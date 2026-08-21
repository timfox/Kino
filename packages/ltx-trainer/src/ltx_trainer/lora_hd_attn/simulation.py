"""Finite-D numpy simulation stub (Fig. 1 square markers)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lora_hd_attn.attention import attention_predict
from ltx_trainer.lora_hd_attn.config import LoraHdAttnConfig
from ltx_trainer.lora_hd_attn.losses import mse_fro


def _synthetic_batch(cfg: LoraHdAttnConfig, n: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, ys = [], []
    p0 = min(cfg.P0, max(4, cfg.D // 8))
    w_star = rng.normal(size=(cfg.D, p0)) / np.sqrt(max(p0, 1))
    w_lora = rng.normal(size=cfg.D) / np.sqrt(max(cfg.D, 1))
    for _ in range(n):
        x = rng.normal(size=(cfg.T, cfg.D))
        y = attention_predict(
            x,
            w_star,
            w_lora,
            p_rank=p0,
            sigma=cfg.sigma,
        )
        xs.append(x)
        ys.append(y)
    return xs, ys


def _mean_error(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    w_ext: np.ndarray,
    w_lora: np.ndarray | None,
    cfg: LoraHdAttnConfig,
) -> float:
    if not xs:
        return 0.0
    return float(
        np.mean(
            [
                mse_fro(
                    attention_predict(x, w_ext, w_lora, p_rank=w_ext.shape[1], sigma=cfg.sigma),
                    y,
                    cfg.D,
                )
                for x, y in zip(xs, ys, strict=False)
            ]
        )
    )


def run_finite_d_simulation(cfg: LoraHdAttnConfig, *, seed: int = 0) -> dict[str, Any]:
    """Lightweight finite-D proxy: frozen-random vs rank-one LoRA refinement."""
    rng = np.random.default_rng(seed)
    n_pre = min(cfg.N, 32)
    n_ft = min(cfg.N_prime, 16)
    xs, ys = _synthetic_batch(cfg, n_pre, rng)
    xs_f, ys_f = _synthetic_batch(cfg, n_ft, rng)
    xs_p, ys_p = xs[: max(1, n_pre // 4)], ys[: max(1, n_pre // 4)]

    p = min(cfg.P, max(4, cfg.D // 8))
    w_ext = rng.normal(size=(cfg.D, p)) / np.sqrt(max(p, 1))
    w_zero = np.zeros(cfg.D)
    w_lora = rng.normal(size=cfg.D) / np.sqrt(max(cfg.D, 1))

    frozen_err = _mean_error(xs_f, ys_f, w_ext, w_zero, cfg)
    lora_err = _mean_error(xs_f, ys_f, w_ext, w_lora, cfg)
    test_err = _mean_error(xs_p, ys_p, w_ext, w_lora, cfg)

    return {
        "D": cfg.D,
        "train_error_frozen": round(frozen_err, 4),
        "train_error_finetune": round(lora_err, 4),
        "test_error_finetune": round(test_err, 4),
        "lora_improves_over_frozen": frozen_err > lora_err,
        "w_ext_norm": round(float(np.linalg.norm(w_ext)), 4),
        "w_lora_norm": round(float(np.linalg.norm(w_lora)), 4),
    }
