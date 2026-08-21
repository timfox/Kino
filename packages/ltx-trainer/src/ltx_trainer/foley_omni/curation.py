"""Audiovisual curation pipeline + Bandit post-verification (Sec. 3.1, Appendix A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.foley_omni.config import FoleyOmniConfig


@dataclass(frozen=True)
class ClipFilterThresholds:
    min_resolution_p: int = 480
    min_bitrate_mbps: float = 1.0
    motion_score_min: float = 0.1
    motion_score_max: float = 3.2
    audio_quality_min: float = 0.6
    ib_score_min: float = 0.3
    sync_score_min: float = 0.2


def rms_energy_db(waveform: np.ndarray) -> float:
    """Average RMS energy in dB (Eq. 8)."""
    w = np.asarray(waveform, dtype=np.float64).ravel()
    if w.size == 0:
        return -np.inf
    mean_sq = float(np.mean(w ** 2))
    if mean_sq <= 0.0:
        return -np.inf
    return 10.0 * np.log10(mean_sq)


def bandit_stem_passes(
    waveform: np.ndarray,
    *,
    threshold_db: float = -35.0,
) -> bool:
    """Retain annotation iff stem RMS exceeds threshold (Eq. 9)."""
    return bool(rms_energy_db(waveform) > threshold_db)


def passes_visual_audio_filters(
    *,
    resolution_p: int,
    bitrate_mbps: float,
    motion_score: float,
    audio_quality: float,
    ib_score: float,
    sync_score: float,
    thresholds: ClipFilterThresholds | None = None,
) -> bool:
    """Table 7 joint constraints."""
    t = thresholds or ClipFilterThresholds()
    if resolution_p < t.min_resolution_p:
        return False
    if bitrate_mbps < t.min_bitrate_mbps:
        return False
    if not (t.motion_score_min <= motion_score <= t.motion_score_max):
        return False
    if audio_quality < t.audio_quality_min:
        return False
    if ib_score < t.ib_score_min:
        return False
    if sync_score < t.sync_score_min:
        return False
    return True


def verify_component_labels(
    labels: dict[str, bool],
    stems: dict[str, np.ndarray],
    cfg: FoleyOmniConfig | None = None,
) -> dict[str, bool]:
    """
    Gate Gemini labels with Bandit stem energy.

    ``labels`` keys: words, audio, music (predicted present).
    ``stems`` keys: words, audio, music (separated waveforms).
    """
    cfg = cfg or FoleyOmniConfig()
    verified: dict[str, bool] = {}
    for key in ("words", "audio", "music"):
        if not labels.get(key, False):
            verified[key] = False
            continue
        stem = stems.get(key)
        if stem is None:
            verified[key] = False
            continue
        verified[key] = bandit_stem_passes(stem, threshold_db=cfg.bandit_rms_threshold_db)
    return verified
