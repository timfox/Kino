"""ARC-Forcing adversarial post-training (Sec. 4.2, Eq. 6–7)."""

from __future__ import annotations

import math


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def relativistic_loss(d_real: float, d_fake: float) -> float:
    r"""LR = E[f(D(bx) - D(x))] with f = softplus (Eq. 6), scalar toy."""
    return softplus(d_fake - d_real)


def contrastive_loss(d_matched: float, d_mismatched: float) -> float:
    r"""LC on real music with correct vs permuted prompts (Eq. 7)."""
    return softplus(d_mismatched - d_matched)


def arc_forcing_total(
    d_rollout: float,
    d_real: float,
    *,
    d_matched: float,
    d_mismatched: float,
    lambda_c: float = 1.0,
) -> float:
    """Combined discriminator objective LR + λ·LC."""
    return relativistic_loss(d_real, d_rollout) + lambda_c * contrastive_loss(d_matched, d_mismatched)
