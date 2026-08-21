"""Encode / synthesize / evaluate tactile texture representations."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.baselines import (
    encode_ar,
    encode_mfcc,
    encode_speak,
    synthesize_ar,
    synthesize_mfcc,
    synthesize_speak,
)
from ltx_trainer.tactile_spectral.config import TactileSpectralConfig
from ltx_trainer.tactile_spectral.metrics import (
    band_energy_difference,
    predict_similarity_from_bands,
    spectral_correlation,
)
from ltx_trainer.tactile_spectral.rendering import electrovibration_voltage
from ltx_trainer.tactile_spectral.sbeta import encode_sbeta
from ltx_trainer.tactile_spectral.sslope import encode_sslope
from ltx_trainer.tactile_spectral.synthesis import synthesize_sbeta, synthesize_sslope

RepresentationKind = Literal["ar", "mfcc", "speak", "sbeta", "sslope"]


def encode_texture(
    signal: Tensor,
    kind: RepresentationKind,
    *,
    cfg: TactileSpectralConfig | None = None,
) -> dict[str, Any]:
    """Encode lateral friction signal with chosen representation."""
    cfg = cfg or TactileSpectralConfig()
    sr = cfg.sample_rate_hz
    if kind == "ar":
        return {"kind": kind, "coeffs": encode_ar(signal, order=cfg.ar_order)}
    if kind == "mfcc":
        return {"kind": kind, "coeffs": encode_mfcc(signal, sample_rate=sr, n_coeffs=cfg.mfcc_coeffs)}
    if kind == "speak":
        pf, pa = encode_speak(
            signal,
            sample_rate=sr,
            num_peaks=cfg.num_spectral_peaks,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
        )
        return {"kind": kind, "peak_freqs": pf, "peak_amps": pa}
    if kind == "sbeta":
        enc = encode_sbeta(
            signal,
            sample_rate=sr,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
            num_peaks=cfg.num_spectral_peaks,
        )
        return {"kind": kind, **{k: v for k, v in enc.items() if k != "freqs"}}
    if kind == "sslope":
        enc = encode_sslope(
            signal,
            sample_rate=sr,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
            quant_db=cfg.slope_quant_db_per_decade,
        )
        return {"kind": kind, **{k: v for k, v in enc.items() if k != "freqs"}}
    raise ValueError(f"unknown representation: {kind}")


def synthesize_texture(
    encoded: dict[str, Any],
    length: int,
    *,
    cfg: TactileSpectralConfig | None = None,
) -> Tensor:
    cfg = cfg or TactileSpectralConfig()
    sr = cfg.sample_rate_hz
    kind = encoded["kind"]
    if kind == "ar":
        return synthesize_ar(encoded["coeffs"], length)
    if kind == "mfcc":
        return synthesize_mfcc(
            encoded["coeffs"],
            length,
            sample_rate=sr,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
        )
    if kind == "speak":
        return synthesize_speak(encoded["peak_freqs"], encoded["peak_amps"], length, sample_rate=sr)
    if kind == "sbeta":
        return synthesize_sbeta(
            alpha=float(encoded["alpha"]),
            beta=float(encoded["beta"]),
            scale=float(encoded["scale"]),
            length=length,
            sample_rate=sr,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
        )
    if kind == "sslope":
        return synthesize_sslope(
            ra=int(encoded["ra"]),
            rb=int(encoded["rb"]),
            f_peak=float(encoded["f_peak"]),
            scale=float(encoded["scale"]),
            length=length,
            sample_rate=sr,
            f_low=cfg.f_low_hz,
            f_high=cfg.f_high_hz,
        )
    raise ValueError(f"unknown representation: {kind}")


def evaluate_representation(
    original: Tensor,
    kind: RepresentationKind,
    *,
    cfg: TactileSpectralConfig | None = None,
) -> dict[str, float]:
    """Encode → synthesize → spectral correlation + band-energy similarity proxy."""
    cfg = cfg or TactileSpectralConfig()
    length = original.numel()
    enc = encode_texture(original, kind, cfg=cfg)
    synth = synthesize_texture(enc, length, cfg=cfg)
    corr = spectral_correlation(
        original,
        synth,
        sample_rate=cfg.sample_rate_hz,
        f_low=cfg.f_low_hz,
        f_high=cfg.f_high_hz,
    )
    diff = band_energy_difference(
        original,
        synth,
        sample_rate=cfg.sample_rate_hz,
        n_bands=cfg.critical_bands,
        f_low=cfg.f_low_hz,
        f_high=cfg.f_high_hz,
    )
    rating_proxy = predict_similarity_from_bands(diff)
    return {
        "spectral_correlation": corr,
        "similarity_proxy": rating_proxy,
        "representation": kind,
    }


def demo_synthetic_friction(
    *,
    length: int = 4000,
    cfg: TactileSpectralConfig | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Synthetic lateral friction with low- and high-frequency content."""
    cfg = cfg or TactileSpectralConfig()
    device = device or torch.device("cpu")
    sr = cfg.sample_rate_hz
    t = torch.arange(length, device=device, dtype=torch.float32) / sr
    return (
        0.4 * torch.sin(2 * torch.pi * 80.0 * t)
        + 0.2 * torch.sin(2 * torch.pi * 320.0 * t)
        + 0.05 * torch.randn(length, device=device)
    )


def render_on_display(
    signal: Tensor,
    *,
    cfg: TactileSpectralConfig | None = None,
    gain: float = 1.0,
) -> Tensor:
    cfg = cfg or TactileSpectralConfig()
    return electrovibration_voltage(
        signal,
        gain=gain,
        carrier_hz=cfg.carrier_freq_hz,
        sample_rate=cfg.sample_rate_hz,
    )
