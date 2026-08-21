"""MixFake authenticity combinations and SNR helpers."""

from __future__ import annotations

from enum import Enum


class MixLabel(str, Enum):
    """Foreground/background authenticity quadrants (Table I)."""

    RF_RB = "real_fg_real_bg"
    FF_RB = "fake_fg_real_bg"
    RF_FB = "real_fg_fake_bg"
    FF_FB = "fake_fg_fake_bg"


def foreground_label(mix: MixLabel) -> bool:
    """True if foreground is bona fide."""
    return mix in (MixLabel.RF_RB, MixLabel.RF_FB)


def background_label(mix: MixLabel) -> bool:
    """True if background is bona fide."""
    return mix in (MixLabel.RF_RB, MixLabel.FF_RB)


def snr_gain_linear(snr_db: float) -> float:
    """Convert target SNR (dB) to linear background scale (toy)."""
    return 10.0 ** (snr_db / 20.0)
