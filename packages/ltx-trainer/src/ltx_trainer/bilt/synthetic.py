"""Synthetic liquid phantom spectra (Sec. 2.1)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.bilt.config import SPECTRAL_POINTS, WAVELENGTH_MIN_NM, WAVELENGTH_STEP_NM


def wavelength_grid(points: int = SPECTRAL_POINTS) -> Tensor:
    nm = torch.arange(points, dtype=torch.float32) * WAVELENGTH_STEP_NM + WAVELENGTH_MIN_NM
    return nm


def _intralipid_scattering(n: Tensor) -> Tensor:
    """Reduced scattering μ'_s ∝ λ^(-b), b≈1.5 (Aernouts et al.)."""
    lam_um = n / 1000.0
    return 10.0 * lam_um.pow(-1.5)


def _red_ink_absorption(n: Tensor) -> Tensor:
    """Gaussian absorption peak near 550 nm."""
    return 0.8 * torch.exp(-0.5 * ((n - 550.0) / 25.0) ** 2)


def _black_ink_absorption(n: Tensor) -> Tensor:
    """Broadband flat absorption."""
    return 0.15 + 0.05 * torch.exp(-0.5 * ((n - 520.0) / 80.0) ** 2)


def constituent_spectra(points: int = SPECTRAL_POINTS) -> dict[str, Tensor]:
    n = wavelength_grid(points)
    return {
        "il_scatter": _intralipid_scattering(n),
        "red_abs": _red_ink_absorption(n),
        "black_abs": _black_ink_absorption(n),
        "wavelength_nm": n,
    }


def mix_optical_properties(z: Tensor, spectra: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
    """Linear mixture: z[:,0]=IL, z[:,1]=red, z[:,2]=black."""
    if spectra is None:
        spectra = constituent_spectra()
    il = spectra["il_scatter"].to(z.device)
    red = spectra["red_abs"].to(z.device)
    black = spectra["black_abs"].to(z.device)
    mu_s = z[:, 0:1] * il.unsqueeze(0)
    mu_a = z[:, 1:2] * red.unsqueeze(0) + z[:, 2:3] * black.unsqueeze(0)
    return mu_a, mu_s


def _reflectance_stub(mu_a: Tensor, mu_s: Tensor) -> Tensor:
    """Simplified integrating-sphere forward model (Kubelka-Munk-like stub)."""
    return (mu_s / (mu_a + mu_s + 1e-6)).clamp(0.0, 1.0)


def synthetic_batch(
    batch: int = 8,
    *,
    points: int = SPECTRAL_POINTS,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Returns input reflectance (B,L), μ_a (B,L), μ'_s (B,L), latent z (B,3).
    Concentrations drawn uniformly in [0.05, 1.0].
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    z = 0.05 + 0.95 * torch.rand(batch, 3, generator=gen)
    spectra = constituent_spectra(points)
    mu_a, mu_s = mix_optical_properties(z, spectra)
    x = _reflectance_stub(mu_a, mu_s)
    # normalize inputs to [0,1] per paper
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    x = (x - xmin) / (xmax - xmin + 1e-6)
    return x, mu_a, mu_s, z


def dataset_summary() -> dict:
    return {
        "name": "Liquid phantom (intralipid + red/black ink)",
        "samples": 496,
        "cuvette_mm": 5,
        "spectral_range_nm": [500, 800],
        "step_nm": 2,
        "points": SPECTRAL_POINTS,
        "channels": ["total_transmission", "total_reflection"],
        "train_test_split": "80/20 random",
        "constituents": {
            "0": "intralipid (scatterer)",
            "1": "Modena Red ink (absorber)",
            "2": "Indian Ink (absorber)",
        },
    }
