"""Order parameters Q, M, V and q, m, v — Eqs. (8)–(9)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lora_hd_attn.config import LoraHdAttnConfig


def overlap_ow(m: float, q: float, q0: float = 1.0) -> float:
    """Cosine similarity o_w = |w_hat^T w*| / (||w_hat|| ||w*||) -> m/sqrt(q q0)."""
    denom = max(q * q0, 1e-12)
    return float(abs(m) / np.sqrt(denom))


def pretrain_order_params(cfg: LoraHdAttnConfig) -> dict[str, float]:
    """Simplified fixed-point stub for (Q, M, V) — Result 4."""
    lam = max(cfg.lam, 1e-6)
    alpha = max(cfg.alpha, 1e-6)
    # More pre-training samples + lower lambda -> higher alignment M
    m_raw = np.sqrt(alpha) / (np.sqrt(alpha) + 2.0 * lam)
    M = float(np.clip(m_raw, 0.0, 1.0))
    Q = float(cfg.kappa / (1.0 + lam))
    V = float(1.0 / (1.0 + alpha))
    Q0 = cfg.Q0
    return {"Q0": Q0, "Q": Q, "M": M, "V": V}


def finetune_order_params(
    cfg: LoraHdAttnConfig,
    *,
    pretrain: dict[str, float],
    delta_eff: float,
) -> dict[str, float]:
    """Simplified fixed-point stub for (q, m, v) — Result 5."""
    lam_p = max(cfg.lam_prime, 1e-6)
    alpha_p = max(cfg.alpha_prime, 1e-6)
    noise_penalty = delta_eff / max(pretrain["Q0"], 1e-6)
    m = float(np.clip(alpha_p / (alpha_p + 2.0 * lam_p + noise_penalty), 0.0, 1.0))
    q = float(max(m * m, 1e-6))
    v = float(1.0 / (2.0 * lam_p + 1.0 / alpha_p))
    return {"q0": cfg.q0, "q": q, "m": m, "v": v, "o_w": overlap_ow(m, q, cfg.q0)}


def order_params_card(pre: dict[str, float], fine: dict[str, float]) -> dict[str, Any]:
    return {
        "pretrain": pre,
        "finetune": fine,
        "limits": {"Q0_limit": "1 + kappa0", "q0_limit": 1.0},
    }
