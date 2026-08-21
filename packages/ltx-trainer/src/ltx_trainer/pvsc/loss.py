"""Rate-distortion objective stubs (Eq. 25, 27)."""

from __future__ import annotations


def rate_loss(
    symbol_lengths: list[int],
    rate_map_bits: int,
    hyperprior_bits: int,
    *,
    eta: float = 0.2,
    hyperprior_entropy_bits: float = 0.0,
) -> float:
    """Eq. (25) simplified rate penalty."""
    sym_term = sum(symbol_lengths)
    side_term = rate_map_bits + eta * rate_map_bits
    hyper_term = -eta * hyperprior_entropy_bits + hyperprior_bits
    return sym_term + side_term + hyper_term


def reconstruction_loss_l1(x: float, x_hat: float) -> float:
    return abs(x - x_hat)


def overall_objective(
    rec_losses: list[float],
    per_losses: list[float],
    adv_losses: list[float],
    rate: float,
    *,
    lambda_rec: float = 1.0,
    lambda_per: float = 0.8,
    lambda_adv: float = 0.1,
    omega: float = 1.0,
) -> float:
    """Eq. (27) averaged over branches and time steps (toy scalar form)."""
    n = max(len(rec_losses), 1)
    rd = sum(
        lambda_rec * r + lambda_per * p + lambda_adv * a
        for r, p, a in zip(rec_losses, per_losses, adv_losses, strict=False)
    ) / n
    return omega * rd + rate
