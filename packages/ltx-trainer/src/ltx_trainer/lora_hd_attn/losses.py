"""Train and test error stubs — Eqs. (2)–(5), (19)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.lora_hd_attn.effective_noise import delta_eff_independent


def mse_fro(y_hat: np.ndarray, y: np.ndarray, d: float) -> float:
    return float(np.mean((y_hat - y) ** 2) / max(d, 1.0))


def test_error_pretrain_stub(delta: float, Q0: float, M: float, Q: float, T: int = 3) -> float:
    """Linear-sigma proxy for E(W_hat) from effective channel."""
    de = delta_eff_independent(delta, Q0, M, Q)
    return float(T * T * de / max(T * T, 1))


def test_error_finetune_stub(
    delta_eff: float,
    *,
    m: float,
    q: float,
    q0: float,
    T: int = 3,
    frozen_only: bool = False,
) -> float:
    """Proxy for E'(W_hat, w_hat). frozen_only -> lambda' -> inf, w_hat = 0."""
    if frozen_only:
        return float(T * T * (delta_eff + q0) / max(T * T, 1))
    align_gap = max(0.0, 1.0 - m)
    rank_pen = max(0.0, q - m * m)
    return float(delta_eff + align_gap + 0.5 * rank_pen)


def frozen_baseline(delta_eff: float, q0: float, T: int = 3) -> float:
    """Eq. (79)-(80) lambda' -> inf baseline."""
    return test_error_finetune_stub(delta_eff, m=0.0, q=0.0, q0=q0, T=T, frozen_only=True)
