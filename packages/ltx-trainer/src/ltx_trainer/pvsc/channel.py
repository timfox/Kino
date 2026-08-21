"""Wireless channel and bandwidth metrics (Sec. II-A)."""

from __future__ import annotations

import numpy as np


def awgn_receive(
    symbols: np.ndarray,
    *,
    snr_db: float,
    power: float = 1.0,
) -> np.ndarray:
    """Eq. (1) simplified: y = s + n with AWGN."""
    s = np.asarray(symbols, dtype=np.complex128)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_var = power / snr_linear
    noise = np.sqrt(noise_var / 2.0) * (
        np.random.randn(*s.shape) + 1j * np.random.randn(*s.shape)
    )
    return s + noise


def zf_equalize(received: np.ndarray, channel_est: complex) -> np.ndarray:
    """Eq. (4): ZF equalization with estimated CSI."""
    h = complex(channel_est)
    if abs(h) < 1e-12:
        return received
    return received * np.conj(h) / (abs(h) ** 2)


def channel_bandwidth_ratio(
    channel_uses: int,
    *,
    frames: int,
    height: int,
    width: int,
    channels: int = 3,
) -> float:
    """Eq. (5): CBR = sum(l_t) / (T * H * W * C)."""
    n = frames * height * width * channels
    return channel_uses / n if n else 0.0


def nmse_db(estimation_error_power: float, channel_power: float) -> float:
    """Eq. (3): CSI NMSE in dB."""
    if channel_power <= 0:
        return float("inf")
    return 10.0 * np.log10(estimation_error_power / channel_power)
