"""SKILD: scale-invariant k-space diffusion (Chen et al. arXiv:2605.26032).

Unifies unconditional generation and continuous super-resolution via a frequency-space
forward process with spectrum-matched noise. See :mod:`ltx_trainer.skild.inference`.
"""

from ltx_trainer.skild.config import SkildConfig, SkildScheduleConfig
from ltx_trainer.skild.inference import (
    SkildSpectrum,
    continuous_super_resolve,
    estimate_spectrum_from_images,
    generation_from_noise,
    load_spectrum,
    save_spectrum,
    start_timestep_for_scale,
)
from ltx_trainer.skild.spectrum import fit_power_law_spectrum, radial_variance_spectrum

__all__ = [
    "SkildConfig",
    "SkildScheduleConfig",
    "SkildSpectrum",
    "continuous_super_resolve",
    "estimate_spectrum_from_images",
    "fit_power_law_spectrum",
    "generation_from_noise",
    "load_spectrum",
    "radial_variance_spectrum",
    "save_spectrum",
    "start_timestep_for_scale",
]
