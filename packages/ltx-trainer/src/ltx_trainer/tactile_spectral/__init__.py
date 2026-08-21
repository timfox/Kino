"""Compact spectral tactile texture models (Balasubramanian & Vardar, arXiv:2605.23804)."""

from ltx_trainer.tactile_spectral.baselines import encode_ar, encode_mfcc, encode_speak
from ltx_trainer.tactile_spectral.config import (
    REPRESENTATION_NAMES,
    TEXTURE_NAMES,
    TactileSpectralConfig,
)
from ltx_trainer.tactile_spectral.metrics import (
    band_energy_difference,
    critical_band_energies,
    predict_similarity_from_bands,
    spectral_correlation,
)
from ltx_trainer.tactile_spectral.pipeline import (
    demo_synthetic_friction,
    encode_texture,
    evaluate_representation,
    render_on_display,
    synthesize_texture,
)
from ltx_trainer.tactile_spectral.rendering import electrovibration_voltage
from ltx_trainer.tactile_spectral.sbeta import encode_sbeta, fit_sbeta_params, sbeta_envelope
from ltx_trainer.tactile_spectral.sslope import encode_sslope, estimate_sslope_orders, sslope_envelope

__all__ = [
    "REPRESENTATION_NAMES",
    "TEXTURE_NAMES",
    "TactileSpectralConfig",
    "band_energy_difference",
    "critical_band_energies",
    "demo_synthetic_friction",
    "electrovibration_voltage",
    "encode_ar",
    "encode_mfcc",
    "encode_sbeta",
    "encode_speak",
    "encode_sslope",
    "encode_texture",
    "estimate_sslope_orders",
    "evaluate_representation",
    "fit_sbeta_params",
    "predict_similarity_from_bands",
    "render_on_display",
    "sbeta_envelope",
    "spectral_correlation",
    "sslope_envelope",
    "synthesize_texture",
]
