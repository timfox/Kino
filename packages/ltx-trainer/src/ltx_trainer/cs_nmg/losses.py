"""WCE anchor + InfoNCE contrastive alignment (§4)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.cs_nmg.config import CsNmgConfig


def poi_token_weights(n_tokens: int, poi_indices: list[int], *, alpha_wce: float) -> np.ndarray:
    """wt = 1 + (α − 1) mt (Eq. 7)."""
    w = np.ones(n_tokens, dtype=np.float64)
    for i in poi_indices:
        if 0 <= i < n_tokens:
            w[i] = alpha_wce
    return w


def weighted_ce_loss(token_nll: np.ndarray, weights: np.ndarray) -> float:
    """LWCE = −(1/Σwt) Σ wt log p (Eq. 8)."""
    denom = float(np.sum(weights))
    if denom <= 0:
        return 0.0
    return float(-np.sum(weights * token_nll) / denom)


def length_normalized_score(token_logprobs: list[float]) -> float:
    """Sθ(y;x) = (1/|y|) Σ log p (Eq. 9)."""
    if not token_logprobs:
        return 0.0
    return float(sum(token_logprobs) / len(token_logprobs))


def infonce_loss(
    anchor_score: float,
    negative_scores: list[float],
    *,
    beta: float = 1.0,
) -> float:
    """InfoNCE ranking y* above K negatives (Eq. 10)."""
    pos = math.exp(beta * anchor_score)
    denom = pos + sum(math.exp(beta * s) for s in negative_scores)
    if denom <= 0:
        return 0.0
    return float(-math.log(pos / denom))


def combined_loss(
    wce: float,
    cl: float,
    *,
    lambda_cl: float,
) -> float:
    """L = LWCE + λCL LCL (Eq. 11)."""
    return wce + lambda_cl * cl


def losses_demo(*, seed: int = 0, cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CsNmgConfig()
    n = 12
    poi = [3, 4, 5]
    w = poi_token_weights(n, poi, alpha_wce=cfg.alpha_wce_vie)
    ref_nll = np.full(n, 0.1)
    nm_nll = ref_nll.copy()
    nm_nll[poi] += 0.35

    wce_ref = weighted_ce_loss(ref_nll, w)
    wce_nm = weighted_ce_loss(nm_nll, w)

    anchor_lp = [-0.2, -0.15, -0.1, -0.05]
    neg_lp = [[-0.25, -0.2, -0.18, -0.12], [-0.3, -0.28, -0.2, -0.15]]
    anchor_s = length_normalized_score(anchor_lp)
    neg_s = [length_normalized_score(n) for n in neg_lp]
    cl = infonce_loss(anchor_s, neg_s, beta=cfg.infonce_beta)
    total = combined_loss(wce_ref, cl, lambda_cl=cfg.lambda_cl)

    return {
        "wce_reference": wce_ref,
        "wce_near_miss": wce_nm,
        "reference_wce_better_than_near_miss": wce_ref > wce_nm,
        "contrastive_loss": cl,
        "combined_loss": total,
        "lambda_cl": cfg.lambda_cl,
        "anchor_beats_negatives": anchor_s > max(neg_s),
    }
