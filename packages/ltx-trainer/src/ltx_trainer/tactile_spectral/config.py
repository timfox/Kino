"""Configuration for compact tactile spectral envelopes (arXiv:2605.23804)."""

from __future__ import annotations

from dataclasses import dataclass, field


TEXTURE_NAMES: tuple[str, ...] = (
    "fine_fabric",
    "coarse_fabric",
    "corrugated_paper",
    "sandpaper",
    "vinyl",
)

REPRESENTATION_NAMES: tuple[str, ...] = (
    "ar",
    "mfcc",
    "speak",
    "sbeta",
    "sslope",
)


@dataclass
class TactileSpectralConfig:
    """Defaults from Materials and Methods (Sec. 4)."""

    sample_rate_hz: float = 20_000.0
    segment_samples: int = 4000
    f_low_hz: float = 20.0
    f_high_hz: float = 1000.0
    num_spectral_peaks: int = 10
    jnd_fraction: float = 0.12
    ar_order: int = 6
    mfcc_coeffs: int = 10
    mel_filters: int = 20
    slope_quant_db_per_decade: float = 20.0
    carrier_freq_hz: float = 7000.0
    critical_bands: int = 9
    max_filter_banks: int = 20
