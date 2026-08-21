"""Voicing detector + voicing-driven quantizer stubs (arXiv:2606.05892)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.vocodec.config import VoCodecConfig


def frame_energy_f0_band(
    frame: np.ndarray,
    *,
    fs: int,
    f0_min: float,
    f0_max: float,
) -> float:
    """Eq. (1)–(2): FFT energy in fundamental-frequency search band."""
    n = len(frame)
    if n == 0:
        return 0.0
    spectrum = np.fft.rfft(frame)
    k_fft = len(spectrum)
    k_min = max(0, int(math.floor(f0_min * k_fft / fs)))
    k_max = min(k_fft - 1, int(math.ceil(f0_max * k_fft / fs)))
    if k_max < k_min:
        return 0.0
    return float(np.sum(np.abs(spectrum[k_min : k_max + 1])))


def voicing_flag_token(energy: float, *, threshold: float) -> int:
    """Eq. (3): dVF ∈ {0, 1}."""
    return 1 if energy > threshold else 0


def detect_voicing_flags(
    waveform: np.ndarray,
    *,
    fs: int = 16000,
    frame_length: int = 512,
    frame_shift: int = 320,
    cfg: VoCodecConfig | None = None,
) -> list[int]:
    """Streamable voicing detector aligned to encoder downsampling rate D."""
    c = cfg or VoCodecConfig()
    flags: list[int] = []
    for start in range(0, max(1, len(waveform) - frame_length + 1), frame_shift):
        frame = waveform[start : start + frame_length]
        energy = frame_energy_f0_band(
            frame,
            fs=fs,
            f0_min=c.f0_min_hz,
            f0_max=c.f0_max_hz,
        )
        flags.append(voicing_flag_token(energy, threshold=c.energy_threshold))
    return flags


def voiced_frame_bits(cfg: VoCodecConfig | None = None) -> float:
    """Sum of log2 codebook sizes for RSVQ path (Eq. 6 voiced term)."""
    c = cfg or VoCodecConfig()
    sq_bits = c.rsvq_sq_count * math.log2(c.codebook_size)
    ivq_bits = c.rsvq_ivq_count * math.log2(c.codebook_size)
    return sq_bits + ivq_bits


def unvoiced_frame_bits(cfg: VoCodecConfig | None = None) -> float:
    """Single SQ codebook + voicing flag is added separately in bitrate()."""
    c = cfg or VoCodecConfig()
    return math.log2(c.codebook_size)


def bitrate_bps(
    *,
    fs: int,
    voiced_ratio: float,
    cfg: VoCodecConfig | None = None,
) -> float:
    """Eq. (6): bits per second at voiced-frame ratio R."""
    c = cfg or VoCodecConfig()
    d = c.downsampling_rate
    r = min(1.0, max(0.0, voiced_ratio))
    voiced = r * voiced_frame_bits(c)
    unvoiced = (1.0 - r) * unvoiced_frame_bits(c)
    flag_bit = 1.0
    return (fs / d) * (voiced + unvoiced + flag_bit)


def bitrate_kbps_simplified(
    *,
    fs: int,
    voiced_ratio: float,
    cfg: VoCodecConfig | None = None,
) -> float:
    """§3.1 simplified: Bitrate = fs/320 * (11 + 20R) [kbps when fs in Hz]."""
    c = cfg or VoCodecConfig()
    r = min(1.0, max(0.0, voiced_ratio))
    bps = (fs / c.downsampling_rate) * (11.0 + 20.0 * r)
    return bps / 1000.0


def quantization_token_layout(voicing_flag: int, cfg: VoCodecConfig | None = None) -> list[str]:
    """Eq. (5): token names produced per frame."""
    c = cfg or VoCodecConfig()
    if voicing_flag == 1:
        sq = [f"dSQ_{i}" for i in range(1, c.rsvq_sq_count + 1)]
        ivq = [f"dIVQ_{j}" for j in range(1, c.rsvq_ivq_count + 1)]
        return sq + ivq
    return ["dSQ_0"]


def mask_based_quantize_batch(
    encoded: np.ndarray,
    voicing_flags: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Eq. (7)–(8): parallel RSVQ/SQ paths with voicing mask (training stub)."""
    gen = rng or np.random.default_rng(0)
    if encoded.ndim != 2:
        raise ValueError("encoded must be (M, B)")
    m, b = encoded.shape
    voiced_mask = np.tile(voicing_flags.reshape(1, b), (m, 1))
    hat_sv = encoded + gen.normal(0, 0.01, encoded.shape)
    hat_su = encoded + gen.normal(0, 0.05, encoded.shape)
    hat_s = hat_sv * voiced_mask + hat_su * (1.0 - voiced_mask)
    return {
        "quantized": hat_s,
        "voiced_frames": int(np.sum(voicing_flags)),
        "unvoiced_frames": int(b - np.sum(voicing_flags)),
        "mask_shape": list(voiced_mask.shape),
    }
