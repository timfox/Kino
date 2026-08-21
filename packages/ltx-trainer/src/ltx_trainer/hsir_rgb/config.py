"""Configuration for pretrained RGB denoiser → hyperspectral restoration (Picone et al., arXiv:2605.24769)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HSIRRgbConfig:
    """Paper defaults: CAVE / Harvard, 31 bands, DRUNet inner c = 3, K = 11 → E ∈ R^{33×31}."""

    paper_arxiv: str = "arXiv:2605.24769"
    num_spectral_bands: int = 31
    inner_channels: int = 3  # RGB or mono inner denoiser width c
    num_projection_groups: int = 11  # K
    patch_size: int = 256
    train_noise_sigmas: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    datasets: tuple[str, ...] = ("CAVE", "Harvard")
    inner_denoiser_name: str = "DRUNet"
    pnp_framework: str = "HQS (DPIR-style parameterization)"
    train_val_test_split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    harvard_rgb_viz_bands: tuple[int, ...] = (24, 14, 4)  # Fig. 1 display bands (1-based in paper)
    hqs_iterations: int = 8
    hqs_mu: float = 2.5

    @property
    def latent_spectral_dim(self) -> int:
        return self.num_projection_groups * self.inner_channels
