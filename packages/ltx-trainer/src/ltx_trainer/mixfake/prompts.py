"""Multi-stream deep prompt injection (Sec. III-B, Eq. 1, 8)."""

from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def texture_gate(mean_psi: float, flux: float, *, w0: float = 0.0, w1: float = 1.0) -> float:
    r"""g = Sigmoid(MLP([Ψ̄; Flux])) — toy linear gate."""
    return sigmoid(w0 * mean_psi + w1 * flux)


def texture_prompt(
    p_tex: float,
    mean_psi: float,
    *,
    gate: float | None = None,
) -> float:
    r"""P̃_tex = LayerNorm(g·P_tex + (1-g)·Ψ̄) (Eq. 8), scalar toy."""
    g = texture_gate(mean_psi, 0.0) if gate is None else gate
    fused = g * p_tex + (1.0 - g) * mean_psi
    return fused  # LayerNorm omitted in scalar stub


def layer_input_width(
    *,
    n_base: int,
    n_freq: int,
    n_tex: int,
    n_hidden: int,
) -> int:
    r"""|X^(i)| = |P_base; P̃_freq; P̃_tex; H^(i)| (Eq. 1)."""
    return n_base + n_freq + n_tex + n_hidden
