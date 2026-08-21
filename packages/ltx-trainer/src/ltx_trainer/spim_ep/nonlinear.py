"""Element-wise nonlinearity ρ(s) for EP energy (Sec. II)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def rho(s: NDArray[np.floating]) -> NDArray[np.floating]:
    """Sine on [-π/2, π/2], saturate to ±1 outside."""
    out = np.sin(s)
    out = np.where(s > math.pi / 2, 1.0, out)
    out = np.where(s < -math.pi / 2, -1.0, out)
    return out


def rho_prime(s: NDArray[np.floating]) -> NDArray[np.floating]:
    """Derivative of ρ; zero in saturated regions."""
    out = np.cos(s)
    out = np.where(np.abs(s) > math.pi / 2, 0.0, out)
    return out
